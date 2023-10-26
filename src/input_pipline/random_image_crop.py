import tensorflow._api.v2.compat.v1 as tf
import cv2
from src.utils.box_utils import area, intersection


def random_image_crop(
        image, boxes, landmarks, labels,
        probability=0.5,
        min_object_covered=0.9,
        aspect_ratio_range=(0.75, 1.33),
        area_range=(0.5, 1.0),
        overlap_thresh=0.3):

    def crop(image, boxes, landmarks, labels):
        image, boxes, landmarks, labels, _ = \
            _random_crop_image(
                image, boxes, landmarks, labels,
                min_object_covered,
                aspect_ratio_range,
                area_range, overlap_thresh
            )
        return image, boxes, landmarks, labels

    do_it = tf.less(tf.random_uniform([]), probability)
    image, boxes, landmarks, labels = \
        tf.cond(
            do_it,
            lambda: crop(image, boxes, landmarks, labels),
            lambda: (image, boxes, landmarks, labels)
        )


    return image, boxes, landmarks, labels


def _random_crop_image(
        image, boxes, landmarks, labels,
        min_object_covered=1,
        aspect_ratio_range=(0.75, 1.33), area_range=(0.5, 1.0),
        overlap_thresh=0.3):
    """Performs random crop. Given the input image and its bounding boxes,
    this op randomly crops a subimage.  Given a user-provided set of input constraints,
    the crop window is resampled until it satisfies these constraints.
    If within 100 trials it is unable to find a valid crop, the original
    image is returned. Both input boxes and returned boxes are in normalized
    form (e.g., lie in the unit square [0, 1]).

    Arguments:
        image: a float tensor with shape [height, width, 3],
            with pixel values varying between [0, 1].
        boxes: a float tensor containing bounding boxes. It has shape
            [num_boxes, 4]. Boxes are in normalized form, meaning
            their coordinates vary between [0, 1].
            Each row is in the form of [ymin, xmin, ymax, xmax].
        min_object_covered: the cropped image must cover at least this fraction of
            at least one of the input bounding boxes.
        aspect_ratio_range: allowed range for aspect ratio of cropped image.
        area_range: allowed range for area ratio between cropped image and the
            original image.
        overlap_thresh: minimum overlap thresh with new cropped
            image to keep the box.
    Returns:
        image: cropped image.
        boxes: remaining boxes.
        keep_ids: indices of remaining boxes in input boxes tensor.
            They are used to get a slice from the 'labels' tensor (if you have one).
            len(keep_ids) = len(boxes).
    """
    with tf.name_scope('random_crop_image'):

        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(boxes, 0),
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=100,
            use_image_if_no_bounding_boxes=True
        )
        begin, size, window = sample_distorted_bounding_box
        image = tf.slice(image, begin, size)
        window = tf.squeeze(window, axis=[0, 1])

        # remove boxes that are completely outside cropped image
        boxes, inside_window_ids = _prune_completely_outside_window(
            boxes, window
        )
        landmarks = tf.gather(landmarks, inside_window_ids)
        #labels = tf.expand_dims(labels, axis=-1)
        labels = tf.gather(labels, inside_window_ids, axis=0)
        # occlude = tf.gather(occlude, inside_window_ids)
        # blur = tf.gather(blur, inside_window_ids)
        # quality  = tf.gather(quality, inside_window_ids)

        # remove boxes that are two much outside image
        boxes, keep_ids = _prune_non_overlapping_boxes(
            boxes, tf.expand_dims(window, 0), overlap_thresh
        )
        landmarks = tf.gather(landmarks, keep_ids)
        labels = tf.gather(labels, keep_ids, axis=0)
        #labels = tf.squeeze(labels, axis=-1)
        # occlude = tf.gather(occlude, keep_ids)
        # blur = tf.gather(blur, keep_ids)
        # quality = tf.gather(quality, keep_ids)

        # change coordinates of the remaining boxes
        boxes, landmarks = _change_coordinate_frame(boxes, landmarks, window)

        keep_ids = tf.gather(inside_window_ids, keep_ids)
        return image, boxes, landmarks, labels, keep_ids


def _prune_completely_outside_window(boxes, window):
    """Prunes bounding boxes that fall completely outside of the given window.
    This function does not clip partially overflowing boxes.

    Arguments:
        boxes: a float tensor with shape [M_in, 4].
        window: a float tensor with shape [4] representing [ymin, xmin, ymax, xmax]
            of the window.
    Returns:
        boxes: a float tensor with shape [M_out, 4] where 0 <= M_out <= M_in.
        valid_indices: a long tensor with shape [M_out] indexing the valid bounding boxes
            in the input 'boxes' tensor.
    """
    with tf.name_scope('prune_completely_outside_window'):

        y_min, x_min, y_max, x_max = tf.split(
            boxes, num_or_size_splits=4, axis=1)
        # they have shape [None, 1]
        win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
        # they have shape []

        coordinate_violations = tf.concat([
            tf.greater_equal(y_min, win_y_max),
            tf.greater_equal(x_min, win_x_max),
            tf.less_equal(y_max, win_y_min), tf.less_equal(x_max, win_x_min)
        ], axis=1)
        valid_indices = tf.squeeze(
            tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))),
            axis=1
        )
        boxes = tf.gather(boxes, valid_indices)
        return boxes, valid_indices


def _prune_non_overlapping_boxes(boxes1, boxes2, min_overlap=0.0):
    """Prunes the boxes in boxes1 that overlap less than thresh with boxes2.
    For each box in boxes1, we want its IOA to be more than min_overlap with
    at least one of the boxes in boxes2. If it does not, we remove it.

    Arguments:
        boxes1: a float tensor with shape [N, 4].
        boxes2: a float tensor with shape [M, 4].
        min_overlap: minimum required overlap between boxes,
            to count them as overlapping.
    Returns:
        boxes: a float tensor with shape [N', 4].
        keep_inds: a long tensor with shape [N'] indexing kept bounding boxes in the
            first input tensor ('boxes1').
    """
    with tf.name_scope('prune_non_overlapping_boxes'):
        ioa = _ioa(boxes2, boxes1)  # [M, N] tensor
        ioa = tf.reduce_max(ioa, axis=0)  # [N] tensor
        keep_bool = tf.greater_equal(ioa, tf.constant(min_overlap))
        keep_inds = tf.squeeze(tf.where(keep_bool), axis=1)
        boxes = tf.gather(boxes1, keep_inds)
        return boxes, keep_inds


def _change_coordinate_frame(boxes, landmarks, window):
    """Change coordinate frame of the boxes to be relative to window's frame.

    Arguments:
        boxes: a float tensor with shape [N, 4].
        window: a float tensor with shape [4].
    Returns:
        a float tensor with shape [N, 4].
    """
    with tf.name_scope('change_coordinate_frame'):

        dist_img_ymin = window[0]
        dist_img_xmin = window[1]
        dist_img_ymax = window[2]
        dist_img_xmax = window[3]
        win_height = dist_img_ymax - dist_img_ymin
        win_width = dist_img_xmax - dist_img_xmin

        # boxes
        ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
        ymin -= dist_img_ymin
        xmin -= dist_img_xmin
        ymax -= dist_img_ymin
        xmax -= dist_img_xmin
        boxes = tf.stack([
            ymin / win_height, xmin / win_width,
            ymax / win_height, xmax / win_width
        ], axis=1)
        boxes = tf.clip_by_value(boxes, clip_value_min=0.0, clip_value_max=1.0)

        # landmarks

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
        landmarks_20_x, landmarks_20_y = tf.unstack(landmarks, axis=1)

        landmarks_0_x -= dist_img_xmin
        landmarks_1_x -= dist_img_xmin
        landmarks_2_x -= dist_img_xmin
        landmarks_3_x -= dist_img_xmin
        landmarks_4_x -= dist_img_xmin
        landmarks_5_x -= dist_img_xmin
        landmarks_6_x -= dist_img_xmin
        landmarks_7_x -= dist_img_xmin
        landmarks_8_x -= dist_img_xmin
        landmarks_9_x -= dist_img_xmin
        landmarks_10_x -= dist_img_xmin
        landmarks_11_x -= dist_img_xmin
        landmarks_12_x -= dist_img_xmin
        landmarks_13_x -= dist_img_xmin
        landmarks_14_x -= dist_img_xmin
        landmarks_15_x -= dist_img_xmin
        landmarks_16_x -= dist_img_xmin
        landmarks_17_x -= dist_img_xmin
        landmarks_18_x -= dist_img_xmin
        landmarks_19_x -= dist_img_xmin
        landmarks_20_x -= dist_img_xmin


        landmarks_0_y -= dist_img_ymin
        landmarks_1_y -= dist_img_ymin
        landmarks_2_y -= dist_img_ymin
        landmarks_3_y -= dist_img_ymin
        landmarks_4_y -= dist_img_ymin
        landmarks_5_y -= dist_img_ymin
        landmarks_6_y -= dist_img_ymin
        landmarks_7_y -= dist_img_ymin
        landmarks_8_y -= dist_img_ymin
        landmarks_9_y -= dist_img_ymin
        landmarks_10_y -= dist_img_ymin
        landmarks_11_y -= dist_img_ymin
        landmarks_12_y -= dist_img_ymin
        landmarks_13_y -= dist_img_ymin
        landmarks_14_y -= dist_img_ymin
        landmarks_15_y -= dist_img_ymin
        landmarks_16_y -= dist_img_ymin
        landmarks_17_y -= dist_img_ymin
        landmarks_18_y -= dist_img_ymin
        landmarks_19_y -= dist_img_ymin
        landmarks_20_y -= dist_img_ymin



        landmarks = tf.stack([
            landmarks_0_x / win_width,
            landmarks_0_y / win_height,
            landmarks_1_x / win_width,
            landmarks_1_y / win_height,
            landmarks_2_x / win_width,
            landmarks_2_y / win_height,
            landmarks_3_x / win_width,
            landmarks_3_y / win_height,
            landmarks_4_x / win_width,
            landmarks_4_y / win_height,
            landmarks_5_x / win_width,
            landmarks_5_y / win_height,
            landmarks_6_x / win_width,
            landmarks_6_y / win_height,
            landmarks_7_x / win_width,
            landmarks_7_y / win_height,
            landmarks_8_x / win_width,
            landmarks_8_y / win_height,
            landmarks_9_x / win_width,
            landmarks_9_y / win_height,
            landmarks_10_x / win_width,
            landmarks_10_y / win_height,
            landmarks_11_x / win_width,
            landmarks_11_y / win_height,
            landmarks_12_x / win_width,
            landmarks_12_y / win_height,
            landmarks_13_x / win_width,
            landmarks_13_y / win_height,
            landmarks_14_x / win_width,
            landmarks_14_y / win_height,
            landmarks_15_x / win_width,
            landmarks_15_y / win_height,
            landmarks_16_x / win_width,
            landmarks_16_y / win_height,
            landmarks_17_x / win_width,
            landmarks_17_y / win_height,
            landmarks_18_x / win_width,
            landmarks_18_y / win_height,
            landmarks_19_x / win_width,
            landmarks_19_y / win_height,
            landmarks_20_x / win_width,
            landmarks_20_y / win_height,
        ], axis=1)
        landmarks = tf.clip_by_value(landmarks, clip_value_min=0.0, clip_value_max=1.0)
        '''
        lefteye_x, lefteye_y, righteye_x, righteye_y, \
            nose_x, nose_y, \
            leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y = \
            tf.unstack(landmarks, axis=1)

        lefteye_x -= dist_img_xmin
        righteye_x -= dist_img_xmin
        nose_x -= dist_img_xmin
        leftmouth_x -= dist_img_xmin
        rightmouth_x -= dist_img_xmin

        lefteye_y -= dist_img_ymin
        righteye_y -= dist_img_ymin
        nose_y -= dist_img_ymin
        leftmouth_y -= dist_img_ymin
        rightmouth_y -= dist_img_ymin

        landmarks = tf.stack([
            lefteye_x / win_width,
            lefteye_y / win_height,
            righteye_x / win_width,
            righteye_y / win_height,
            nose_x / win_width,
            nose_y / win_height,
            leftmouth_x / win_width,
            leftmouth_y / win_height,
            rightmouth_x / win_width,
            rightmouth_y / win_height
        ], axis=1)
        
        landmarks = tf.clip_by_value(
            landmarks, clip_value_min=0.0, clip_value_max=1.0)
        '''

        return boxes, landmarks


def _ioa(boxes1, boxes2):
    """Computes pairwise intersection-over-area between box collections.
    intersection-over-area (IOA) between two boxes box1 and box2 is defined as
    their intersection area over box2's area. Note that ioa is not symmetric,
    that is, ioa(box1, box2) != ioa(box2, box1).

    Arguments:
        boxes1: a float tensor with shape [N, 4].
        boxes2: a float tensor with shape [M, 4].
    Returns:
        a float tensor with shape [N, M] representing pairwise ioa scores.
    """
    with tf.name_scope('ioa'):
        intersections = intersection(boxes1, boxes2)  # shape [N, M]
        areas = tf.expand_dims(area(boxes2), 0)  # shape [1, M]
        return tf.divide(intersections, areas)
