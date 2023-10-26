import tensorflow._api.v2.compat.v1 as tf
from src.constants import EPSILON, SCALE_FACTORS, SCALE_FACTOR_LANDMARK
"""
Tools for dealing with bounding boxes.
All boxes are of the format [ymin, xmin, ymax, xmax]
if not stated otherwise.
And box coordinates are normalized to [0, 1] range.
"""


def choose_anchors_center_in_gt(groundtruth_boxes, anchors):
    """
    choose center of anchors in the gt
    :param groundtruth_boxes:
    :param anchors:
    :return:  shape [N, num_anchors]
    """
    #ymin, xmin, ymax, xmax = tf.split(groundtruth_boxes, num_or_size_splits=4, axis=1)
    ymin, xmin, ymax, xmax = tf.unstack(groundtruth_boxes, axis=1)
    ycenter_a, xcenter_a, _, _ = to_center_coordinates(tf.unstack(anchors, axis=1))
    mask = tf.less(tf.expand_dims(ymin, 1) - tf.expand_dims(ycenter_a, 0), 0)  # [N, num_anchors]
    mask &= tf.less(tf.expand_dims(xmin, 1) - tf.expand_dims(xcenter_a, 0), 0)  # [N, num_anchors]
    mask &= tf.greater(tf.expand_dims(ymax, 1) - tf.expand_dims(ycenter_a, 0), 0)  # [N, num_anchors]
    mask &= tf.greater(tf.expand_dims(xmax, 1) - tf.expand_dims(xcenter_a, 0), 0)  # [N, num_anchors]
    return mask



def calc_iou_mean_std(similarity_matrix):
    """
    :param similarity_matrix: shape [N, num_achors]
    :return: mean_iou, std_iou  shape [N]
    """
    s = tf.reduce_sum(similarity_matrix, axis=1)  # shape [N]
    mask = tf.to_float(tf.greater(similarity_matrix, 0))
    n = tf.reduce_sum(mask, axis=1)

    mean_iou = tf.divide(s, (n + EPSILON))

    mean_iou_tiled = tf.expand_dims(mean_iou, axis=1)
    num_anchors = tf.shape(similarity_matrix)[1]
    mean_iou_tiled = tf.tile(mean_iou_tiled, [1, num_anchors])
    mean_iou_tiled *= mask

    std_iou = tf.reduce_sum(tf.square(similarity_matrix - mean_iou_tiled), axis=1)
    std_iou = tf.divide(std_iou, (n + EPSILON))
    std_iou = tf.math.sqrt(std_iou)

    return mean_iou, std_iou



def get_top_k_mask(center_distance, k):
    """
    for every groundtruth choose top k anchors
    :param center_distance:  shape [N, num_anchors]
    :return: mask: shape [N, num_anchors]
    """
    center_distance = -center_distance
    values, indices = tf.nn.top_k(center_distance, k=k)
    range_i = tf.range(tf.shape(indices)[0])
    range_j = tf.range(tf.shape(indices)[1])
    ii, jj = tf.meshgrid(range_j, range_i)
    indices = tf.stack([jj, indices], axis=-1)
    update = tf.ones(tf.shape(indices)[:-1])
    shape = tf.shape(center_distance)
    mask = tf.scatter_nd(indices, update, shape)
    mask = tf.greater(mask, 0)

    #min_values = tf.reduce_min(values, axis=-1)
    #mask = tf.greater(tf.transpose(center_distance), min_values - EPSILON)
    #mask = tf.transpose(mask)

    return mask


def L2_distance(boxes1, boxes2):
    with tf.name_scope('L2_distance'):
        ycenter_1, xcenter_1, _, _ = to_center_coordinates(tf.unstack(boxes1, axis=1))
        ycenter_2, xcenter_2, _, _ = to_center_coordinates(tf.unstack(boxes2, axis=1))
        y_square = tf.square(tf.expand_dims(ycenter_1, 1) - tf.expand_dims(ycenter_2, 0))
        x_square = tf.square(tf.expand_dims(xcenter_1, 1) - tf.expand_dims(xcenter_2, 0))
        dist = x_square + y_square
    return dist


def iou(boxes1, boxes2):
    """Computes pairwise intersection-over-union between two box collections.

    Arguments:
        boxes1: a float tensor with shape [N, 4].
        boxes2: a float tensor with shape [M, 4].
    Returns:
        a float tensor with shape [N, M] representing pairwise iou scores.
    """
    with tf.name_scope('iou'):
        intersections = intersection(boxes1, boxes2)
        areas1 = area(boxes1)
        areas2 = area(boxes2)
        unions = tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections
        return tf.clip_by_value(tf.divide(intersections, unions), 0.0, 1.0)


def intersection(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.

    Arguments:
        boxes1: a float tensor with shape [N, 4].
        boxes2: a float tensor with shape [M, 4].
    Returns:
        a float tensor with shape [N, M] representing pairwise intersections.
    """
    with tf.name_scope('intersection'):

        ymin1, xmin1, ymax1, xmax1 = tf.split(boxes1, num_or_size_splits=4, axis=1)
        ymin2, xmin2, ymax2, xmax2 = tf.split(boxes2, num_or_size_splits=4, axis=1)
        # they all have shapes like [None, 1]

        all_pairs_min_ymax = tf.minimum(ymax1, tf.transpose(ymax2))
        all_pairs_max_ymin = tf.maximum(ymin1, tf.transpose(ymin2))
        intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(xmax1, tf.transpose(xmax2))
        all_pairs_max_xmin = tf.maximum(xmin1, tf.transpose(xmin2))
        intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        # they all have shape [N, M]

        return intersect_heights * intersect_widths


def area(boxes):
    """Computes area of boxes.

    Arguments:
        boxes: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N] representing box areas.
    """
    with tf.name_scope('area'):
        ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
        return (ymax - ymin) * (xmax - xmin)


def to_minmax_coordinates(boxes):
    """Convert bounding boxes of the format
    [cy, cx, h, w] to the format [ymin, xmin, ymax, xmax].

    Arguments:
        boxes: a list of float tensors with shape [N]
            that represent cy, cx, h, w.
    Returns:
        a list of float tensors with shape [N]
        that represent ymin, xmin, ymax, xmax.
    """
    with tf.name_scope('to_minmax_coordinates'):
        cy, cx, h, w = boxes
        ymin, xmin = cy - 0.5*h, cx - 0.5*w
        ymax, xmax = cy + 0.5*h, cx + 0.5*w
        return [ymin, xmin, ymax, xmax]


def to_center_coordinates(boxes):
    """Convert bounding boxes of the format
    [ymin, xmin, ymax, xmax] to the format [cy, cx, h, w].

    Arguments:
        boxes: a list of float tensors with shape [N]
            that represent ymin, xmin, ymax, xmax.
    Returns:
        a list of float tensors with shape [N]
        that represent cy, cx, h, w.
    """
    with tf.name_scope('to_center_coordinates'):
        ymin, xmin, ymax, xmax = boxes
        h = ymax - ymin
        w = xmax - xmin
        cy = ymin + 0.5*h
        cx = xmin + 0.5*w
        return [cy, cx, h, w]


def encode_boxes(boxes, anchors):
    """Encode boxes with respect to anchors.

    Arguments:
        boxes: a float tensor with shape [N, 4].
        anchors: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N, 4],
        anchor-encoded boxes of the format [ty, tx, th, tw].
    """
    with tf.name_scope('encode_boxes_groundtruth'):

        ycenter_a, xcenter_a, ha, wa = to_center_coordinates(tf.unstack(anchors, axis=1))
        ycenter, xcenter, h, w = to_center_coordinates(tf.unstack(boxes, axis=1))

        # to avoid NaN in division and log below
        ha += EPSILON
        wa += EPSILON
        h += EPSILON
        w += EPSILON

        tx = (xcenter - xcenter_a)/wa
        ty = (ycenter - ycenter_a)/ha
        tw = tf.log(w / wa)
        th = tf.log(h / ha)

        ty *= SCALE_FACTORS[0]
        tx *= SCALE_FACTORS[1]
        th *= SCALE_FACTORS[2]
        tw *= SCALE_FACTORS[3]

        return tf.stack([ty, tx, th, tw], axis=1)

def encode_landmarks(landmarks, anchors):
    """Encode boxes with respect to anchors.

    Arguments:
        boxes: a float tensor with shape [N, 4].
        anchors: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N, 4],
        anchor-encoded boxes of the format [ty, tx, th, tw].
    """
    with tf.name_scope('encode_landmarks_groundtruth'):

        ycenter_a, xcenter_a, ha, wa = to_center_coordinates(tf.unstack(anchors, axis=1))
        #ycenter, xcenter, h, w = to_center_coordinates(tf.unstack(boxes, axis=1))
        landmarks_0_x, landmarks_0_y, \
        landmarks_1_x, landmarks_1_y, \
        landmarks_2_x, landmarks_2_y, \
        landmarks_3_x, landmarks_3_y, \
        landmarks_4_x, landmarks_4_y, \
        landmarks_5_x, landmarks_5_y, \
        landmarks_6_x, landmarks_6_y, \
        landmarks_7_x, landmarks_7_y, \
        landmarks_8_x, landmarks_8_y, \
        landmarks_9_x, landmarks_9_y, \
        landmarks_10_x, landmarks_10_y, \
        landmarks_11_x, landmarks_11_y, \
        landmarks_12_x, landmarks_12_y, \
        landmarks_13_x, landmarks_13_y, \
        landmarks_14_x, landmarks_14_y, \
        landmarks_15_x, landmarks_15_y, \
        landmarks_16_x, landmarks_16_y, \
        landmarks_17_x, landmarks_17_y, \
        landmarks_18_x, landmarks_18_y, \
        landmarks_19_x, landmarks_19_y, \
        landmarks_20_x, landmarks_20_y = \
            tf.unstack(landmarks, axis=1)

        # to avoid NaN in division
        ha += EPSILON
        wa += EPSILON

        t_landmarks_0_x = (landmarks_0_x    - xcenter_a) / wa
        t_landmarks_1_x = (landmarks_1_x    - xcenter_a) / wa
        t_landmarks_2_x = (landmarks_2_x    - xcenter_a) / wa
        t_landmarks_3_x = (landmarks_3_x    - xcenter_a) / wa
        t_landmarks_4_x = (landmarks_4_x    - xcenter_a) / wa
        t_landmarks_5_x = (landmarks_5_x    - xcenter_a) / wa
        t_landmarks_6_x = (landmarks_6_x    - xcenter_a) / wa
        t_landmarks_7_x = (landmarks_7_x    - xcenter_a) / wa
        t_landmarks_8_x = (landmarks_8_x    - xcenter_a) / wa
        t_landmarks_9_x = (landmarks_9_x    - xcenter_a) / wa
        t_landmarks_10_x = (landmarks_10_x    - xcenter_a) / wa
        t_landmarks_11_x = (landmarks_11_x    - xcenter_a) / wa
        t_landmarks_12_x = (landmarks_12_x    - xcenter_a) / wa
        t_landmarks_13_x = (landmarks_13_x    - xcenter_a) / wa
        t_landmarks_14_x = (landmarks_14_x    - xcenter_a) / wa
        t_landmarks_15_x = (landmarks_15_x    - xcenter_a) / wa
        t_landmarks_16_x = (landmarks_16_x    - xcenter_a) / wa
        t_landmarks_17_x = (landmarks_17_x    - xcenter_a) / wa
        t_landmarks_18_x = (landmarks_18_x    - xcenter_a) / wa
        t_landmarks_19_x = (landmarks_19_x    - xcenter_a) / wa
        t_landmarks_20_x = (landmarks_20_x    - xcenter_a) / wa

        t_landmarks_0_y = (landmarks_0_y    - ycenter_a) / ha
        t_landmarks_1_y = (landmarks_1_y    - ycenter_a) / ha
        t_landmarks_2_y = (landmarks_2_y    - ycenter_a) / ha
        t_landmarks_3_y = (landmarks_3_y    - ycenter_a) / ha
        t_landmarks_4_y = (landmarks_4_y    - ycenter_a) / ha
        t_landmarks_5_y = (landmarks_5_y    - ycenter_a) / ha
        t_landmarks_6_y = (landmarks_6_y    - ycenter_a) / ha
        t_landmarks_7_y = (landmarks_7_y    - ycenter_a) / ha
        t_landmarks_8_y = (landmarks_8_y    - ycenter_a) / ha
        t_landmarks_9_y = (landmarks_9_y    - ycenter_a) / ha
        t_landmarks_10_y = (landmarks_10_y    - ycenter_a) / ha
        t_landmarks_11_y = (landmarks_11_y    - ycenter_a) / ha
        t_landmarks_12_y = (landmarks_12_y    - ycenter_a) / ha
        t_landmarks_13_y = (landmarks_13_y    - ycenter_a) / ha
        t_landmarks_14_y = (landmarks_14_y    - ycenter_a) / ha
        t_landmarks_15_y = (landmarks_15_y    - ycenter_a) / ha
        t_landmarks_16_y = (landmarks_16_y    - ycenter_a) / ha
        t_landmarks_17_y = (landmarks_17_y    - ycenter_a) / ha
        t_landmarks_18_y = (landmarks_18_y    - ycenter_a) / ha
        t_landmarks_19_y = (landmarks_19_y    - ycenter_a) / ha
        t_landmarks_20_y = (landmarks_20_y    - ycenter_a) / ha

        t_landmarks_0_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_1_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_2_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_3_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_4_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_5_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_6_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_7_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_8_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_9_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_10_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_11_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_12_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_13_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_14_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_15_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_16_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_17_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_18_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_19_x *= SCALE_FACTOR_LANDMARK
        t_landmarks_20_x *= SCALE_FACTOR_LANDMARK

        t_landmarks_0_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_1_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_2_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_3_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_4_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_5_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_6_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_7_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_8_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_9_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_10_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_11_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_12_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_13_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_14_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_15_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_16_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_17_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_18_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_19_y *= SCALE_FACTOR_LANDMARK
        t_landmarks_20_y *= SCALE_FACTOR_LANDMARK

        landmarks_gt_codes = tf.stack([
            t_landmarks_0_x, t_landmarks_0_y,
            t_landmarks_1_x, t_landmarks_1_y,
            t_landmarks_2_x, t_landmarks_2_y,
            t_landmarks_3_x, t_landmarks_3_y,
            t_landmarks_4_x, t_landmarks_4_y,
            t_landmarks_5_x, t_landmarks_5_y,
            t_landmarks_6_x, t_landmarks_6_y,
            t_landmarks_7_x, t_landmarks_7_y,
            t_landmarks_8_x, t_landmarks_8_y,
            t_landmarks_9_x, t_landmarks_9_y,
            t_landmarks_10_x, t_landmarks_10_y,
            t_landmarks_11_x, t_landmarks_11_y,
            t_landmarks_12_x, t_landmarks_12_y,
            t_landmarks_13_x, t_landmarks_13_y,
            t_landmarks_14_x, t_landmarks_14_y,
            t_landmarks_15_x, t_landmarks_15_y,
            t_landmarks_16_x, t_landmarks_16_y,
            t_landmarks_17_x, t_landmarks_17_y,
            t_landmarks_18_x, t_landmarks_18_y,
            t_landmarks_19_x, t_landmarks_19_y,
            t_landmarks_20_x, t_landmarks_20_y,
        ], axis=1)

        return landmarks_gt_codes


def decode_boxes(codes, anchors):
    """Decode relative codes to boxes.

    Arguments:
        codes: a float tensor with shape [N, 4],
            anchor-encoded boxes of the format [ty, tx, th, tw].
        anchors: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N, 4],
        bounding boxes of the format [ymin, xmin, ymax, xmax].
    """
    with tf.name_scope('decode_boxes_predictions'):

        ycenter_a, xcenter_a, ha, wa = to_center_coordinates(tf.unstack(anchors, axis=1))
        ty, tx, th, tw = tf.unstack(codes, axis=1)

        ty /= SCALE_FACTORS[0]
        tx /= SCALE_FACTORS[1]
        th /= SCALE_FACTORS[2]
        tw /= SCALE_FACTORS[3]
        w = tf.exp(tw) * wa
        h = tf.exp(th) * ha
        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a

        return tf.stack(to_minmax_coordinates([ycenter, xcenter, h, w]), axis=1)

def decode_landmarks(codes, anchors):
    """Decode relative codes to boxes.

    Arguments:
        codes: a float tensor with shape [N, 10],
            anchor-encoded boxes of the format [ty, tx, th, tw].
        anchors: a float tensor with shape [N, 4].
    Returns:
        a float tensor with shape [N, 10],
        landmarks of the format [lefteye_x, ...]
    """
    with tf.name_scope('decode_landmarks_predictions'):

        ycenter_a, xcenter_a, ha, wa = to_center_coordinates(tf.unstack(anchors, axis=1))
        #ty, tx, th, tw = tf.unstack(codes, axis=1)
        t_landmarks_0_x, t_landmarks_0_y, \
        t_landmarks_1_x, t_landmarks_1_y, \
        t_landmarks_2_x, t_landmarks_2_y, \
        t_landmarks_3_x, t_landmarks_3_y, \
        t_landmarks_4_x, t_landmarks_4_y, \
        t_landmarks_5_x, t_landmarks_5_y, \
        t_landmarks_6_x, t_landmarks_6_y, \
        t_landmarks_7_x, t_landmarks_7_y, \
        t_landmarks_8_x, t_landmarks_8_y, \
        t_landmarks_9_x, t_landmarks_9_y, \
        t_landmarks_10_x, t_landmarks_10_y, \
        t_landmarks_11_x, t_landmarks_11_y, \
        t_landmarks_12_x, t_landmarks_12_y, \
        t_landmarks_13_x, t_landmarks_13_y, \
        t_landmarks_14_x, t_landmarks_14_y, \
        t_landmarks_15_x, t_landmarks_15_y, \
        t_landmarks_16_x, t_landmarks_16_y, \
        t_landmarks_17_x, t_landmarks_17_y, \
        t_landmarks_18_x, t_landmarks_18_y, \
        t_landmarks_19_x, t_landmarks_19_y, \
        t_landmarks_20_x, t_landmarks_20_y, = \
            tf.unstack(codes, axis=1)

        t_landmarks_0_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_1_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_2_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_3_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_4_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_5_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_6_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_7_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_8_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_9_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_10_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_11_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_12_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_13_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_14_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_15_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_16_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_17_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_18_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_19_x /= SCALE_FACTOR_LANDMARK
        t_landmarks_20_x /= SCALE_FACTOR_LANDMARK

        t_landmarks_0_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_1_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_2_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_3_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_4_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_5_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_6_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_7_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_8_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_9_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_10_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_11_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_12_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_13_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_14_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_15_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_16_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_17_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_18_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_19_y /= SCALE_FACTOR_LANDMARK
        t_landmarks_20_y /= SCALE_FACTOR_LANDMARK

        landmarks_0_x = t_landmarks_0_x * wa + xcenter_a
        landmarks_1_x = t_landmarks_1_x * wa + xcenter_a
        landmarks_2_x = t_landmarks_2_x * wa + xcenter_a
        landmarks_3_x = t_landmarks_3_x * wa + xcenter_a
        landmarks_4_x = t_landmarks_4_x * wa + xcenter_a
        landmarks_5_x = t_landmarks_5_x * wa + xcenter_a
        landmarks_6_x = t_landmarks_6_x * wa + xcenter_a
        landmarks_7_x = t_landmarks_7_x * wa + xcenter_a
        landmarks_8_x = t_landmarks_8_x * wa + xcenter_a
        landmarks_9_x = t_landmarks_9_x * wa + xcenter_a
        landmarks_10_x = t_landmarks_10_x * wa + xcenter_a
        landmarks_11_x = t_landmarks_11_x * wa + xcenter_a
        landmarks_12_x = t_landmarks_12_x * wa + xcenter_a
        landmarks_13_x = t_landmarks_13_x * wa + xcenter_a
        landmarks_14_x = t_landmarks_14_x * wa + xcenter_a
        landmarks_15_x = t_landmarks_15_x * wa + xcenter_a
        landmarks_16_x = t_landmarks_16_x * wa + xcenter_a
        landmarks_17_x = t_landmarks_17_x * wa + xcenter_a
        landmarks_18_x = t_landmarks_18_x * wa + xcenter_a
        landmarks_19_x = t_landmarks_19_x * wa + xcenter_a
        landmarks_20_x = t_landmarks_20_x * wa + xcenter_a

        landmarks_0_y = t_landmarks_0_y * ha + ycenter_a
        landmarks_1_y = t_landmarks_1_y * ha + ycenter_a
        landmarks_2_y = t_landmarks_2_y * ha + ycenter_a
        landmarks_3_y = t_landmarks_3_y * ha + ycenter_a
        landmarks_4_y = t_landmarks_4_y * ha + ycenter_a
        landmarks_5_y = t_landmarks_5_y * ha + ycenter_a
        landmarks_6_y = t_landmarks_6_y * ha + ycenter_a
        landmarks_7_y = t_landmarks_7_y * ha + ycenter_a
        landmarks_8_y = t_landmarks_8_y * ha + ycenter_a
        landmarks_9_y = t_landmarks_9_y * ha + ycenter_a
        landmarks_10_y = t_landmarks_10_y * ha + ycenter_a
        landmarks_11_y = t_landmarks_11_y * ha + ycenter_a
        landmarks_12_y = t_landmarks_12_y * ha + ycenter_a
        landmarks_13_y = t_landmarks_13_y * ha + ycenter_a
        landmarks_14_y = t_landmarks_14_y * ha + ycenter_a
        landmarks_15_y = t_landmarks_15_y * ha + ycenter_a
        landmarks_16_y = t_landmarks_16_y * ha + ycenter_a
        landmarks_17_y = t_landmarks_17_y * ha + ycenter_a
        landmarks_18_y = t_landmarks_18_y * ha + ycenter_a
        landmarks_19_y = t_landmarks_19_y * ha + ycenter_a
        landmarks_20_y = t_landmarks_20_y * ha + ycenter_a



        landmarks = tf.stack([
            landmarks_0_x, landmarks_0_y,
            landmarks_1_x, landmarks_1_y,
            landmarks_2_x, landmarks_2_y,
            landmarks_3_x, landmarks_3_y,
            landmarks_4_x, landmarks_4_y,
            landmarks_5_x, landmarks_5_y,
            landmarks_6_x, landmarks_6_y,
            landmarks_7_x, landmarks_7_y,
            landmarks_8_x, landmarks_8_y,
            landmarks_9_x, landmarks_9_y,
            landmarks_10_x, landmarks_10_y,
            landmarks_11_x, landmarks_11_y,
            landmarks_12_x, landmarks_12_y,
            landmarks_13_x, landmarks_13_y,
            landmarks_14_x, landmarks_14_y,
            landmarks_15_x, landmarks_15_y,
            landmarks_16_x, landmarks_16_y,
            landmarks_17_x, landmarks_17_y,
            landmarks_18_x, landmarks_18_y,
            landmarks_19_x, landmarks_19_y,
            landmarks_20_x, landmarks_20_y
        ], axis=1)

        return landmarks


def batch_decode(box_encodings, anchors, landmark_encodings):
    """Decodes a batch of box encodings with respect to the anchors.

    Arguments:
        box_encodings: a float tensor with shape [batch_size, num_anchors, 4].
        anchors: a float tensor with shape [num_anchors, 4].
    Returns:
        a float tensor with shape [batch_size, num_anchors, 4].
        It contains the decoded boxes.
    """
    batch_size = tf.shape(box_encodings)[0]
    num_anchors = tf.shape(box_encodings)[1]

    # boxes
    tiled_anchor_boxes = tf.tile(
        tf.expand_dims(anchors, 0),
        [batch_size, 1, 1]
    )  # shape [batch_size, num_anchors, 4]
    decoded_boxes = decode_boxes(
        tf.reshape(box_encodings, [-1, 4]),
        tf.reshape(tiled_anchor_boxes, [-1, 4])
    )  # shape [batch_size * num_anchors, 4]

    decoded_boxes = tf.reshape(
        decoded_boxes,
        [batch_size, num_anchors, 4]
    )
    decoded_boxes = tf.clip_by_value(decoded_boxes, 0.0, 1.0)
    # '''
    # landmarks
    decoded_landmarks = decode_landmarks(
        tf.reshape(landmark_encodings, [-1, 42]),
        tf.reshape(tiled_anchor_boxes, [-1, 4])
    )  # shape [batch_size * num_anchors, 4]

    decoded_landmarks = tf.reshape(
        decoded_landmarks,
        [batch_size, num_anchors, 42]
    )
    decoded_landmarks = tf.clip_by_value(decoded_landmarks, 0.0, 1.0)
    # '''
    return decoded_boxes, decoded_landmarks


def batch_decode_box(box_encodings, anchors):
    """Decodes a batch of box encodings with respect to the anchors.

    Arguments:
        box_encodings: a float tensor with shape [batch_size, num_anchors, 4].
        anchors: a float tensor with shape [num_anchors, 4].
    Returns:
        a float tensor with shape [batch_size, num_anchors, 4].
        It contains the decoded boxes.
    """
    batch_size = tf.shape(box_encodings)[0]
    num_anchors = tf.shape(box_encodings)[1]

    # boxes
    tiled_anchor_boxes = tf.tile(
        tf.expand_dims(anchors, 0),
        [batch_size, 1, 1]
    )  # shape [batch_size, num_anchors, 4]
    decoded_boxes = decode_boxes(
        tf.reshape(box_encodings, [-1, 4]),
        tf.reshape(tiled_anchor_boxes, [-1, 4])
    )  # shape [batch_size * num_anchors, 4]

    decoded_boxes = tf.reshape(
        decoded_boxes,
        [batch_size, num_anchors, 4]
    )
    decoded_boxes = tf.clip_by_value(decoded_boxes, 0.0, 1.0)

    return decoded_boxes
