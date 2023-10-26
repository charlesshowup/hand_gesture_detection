import tensorflow._api.v2.compat.v1 as tf
import math
from config import params
from src.utils.box_utils import batch_decode_box, iou, to_center_coordinates

params = params['model_params']
"""
Note that we have only one label (it is 'face'),
so num_classes = 1.
"""


def calc_localization_loss(predictions, targets, weights):
    """A usual L1 smooth loss.

    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, 4],
            representing the (encoded) predicted locations of objects.
        targets: a float tensor with shape [batch_size, num_anchors, 4],
            representing the regression targets.
        weights: a float tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    abs_diff = tf.abs(predictions - targets)
    abs_diff_lt_1 = tf.less(abs_diff, 1.0)
    loss = weights * tf.reduce_sum(
        tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5), axis=2
    )
    return loss


def calc_localization_diou_loss(predictions, targets, anchors, weights):
    """ DIoU loss

    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, 4],
            representing the (encoded) predicted locations of objects.
        targets: a float tensor with shape [batch_size, num_anchors, 4],
            representing the regression targets.
        weights: a float tensor with shape [batch_size, num_anchors].
        anchors: a float tensor with shape [num_anchors, 4].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    batch_size = tf.shape(predictions)[0]
    num_anchors = tf.shape(predictions)[1]

    pred_boxes = batch_decode_box(predictions, anchors)  # shape [batch_size, num_anchors, 4] y1, x1, y2 ,x2
    gt_boxes = targets
    #gt_boxes = batch_decode_box(targets, anchors)  # shape [batch_size, num_anchors, 4]

    pred_boxes = tf.reshape(pred_boxes, [-1, 4])
    gt_boxes = tf.reshape(gt_boxes, [-1, 4])

    diou = calc_ciou(pred_boxes, gt_boxes)
    diou = tf.reshape(diou, [batch_size, num_anchors])
    loss = (1.0 - diou) * weights

    return loss


def calc_diou(pred_boxes, gt_boxes):
    """This is for DIoU"""

    iou_pred_gt = compute_iou_pred_gt(pred_boxes, gt_boxes)
    p_square = calc_p_square(pred_boxes, gt_boxes)
    c_square = calc_c_square(pred_boxes, gt_boxes)

    EPSILON = 1e-14
    diou = iou_pred_gt - p_square / (c_square + EPSILON)  # shape [batch_size * num_anchors]
    diou = tf.clip_by_value(diou, clip_value_min=-1.0, clip_value_max=1.0)

    return diou


def calc_giou(pred_boxes, gt_boxes):
    """This is for GIoU"""

    iou_pred_gt, union = compute_iou_and_union_pred_gt(pred_boxes, gt_boxes)
    c_area = calc_c_area(pred_boxes, gt_boxes)

    EPSILON = 1e-14
    giou = iou_pred_gt - (c_area - union) / (c_area + EPSILON)  # shape [batch_size * num_anchors]
    giou = tf.clip_by_value(giou, clip_value_min=-1.0, clip_value_max=1.0)

    return giou


def calc_ciou(pred_boxes, gt_boxes):
    """This is for DIoU"""

    iou_pred_gt = compute_iou_pred_gt(pred_boxes, gt_boxes)
    p_square = calc_p_square(pred_boxes, gt_boxes)
    c_square = calc_c_square(pred_boxes, gt_boxes)

    EPSILON = 1e-14
    av = compute_av(pred_boxes, gt_boxes, iou_pred_gt)
    ciou = iou_pred_gt - p_square / (c_square + EPSILON) - av  # shape [batch_size * num_anchors]
    ciou = tf.clip_by_value(ciou, clip_value_min=-1.0, clip_value_max=1.0)

    return ciou


def compute_av(pred_boxes, gt_boxes, iou):
    pred_ymin, pred_xmin, pred_ymax, pred_xmax = tf.unstack(pred_boxes, axis=1)
    gt_ymin, gt_xmin, gt_ymax, gt_xmax = tf.unstack(gt_boxes, axis=1)
    w1 = pred_xmax - pred_xmin
    h1 = pred_ymax - pred_ymin
    w2 = gt_xmax - gt_xmin
    h2 = gt_ymax - gt_ymin
    arctan = tf.stop_gradient(tf.atan2(w2, h2) - tf.atan2(w1, h1))
    v      = tf.stop_gradient((4 / (math.pi ** 2)) * tf.square(tf.atan2(w2, h2) - tf.atan2(w1, h1)))
    S      = tf.stop_gradient(1 - iou)
    alpha  = tf.stop_gradient(v / (S + v))
    w_temp = tf.stop_gradient(2 * w1)

    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    return alpha * ar


def compute_iou_pred_gt(pred_boxes, gt_boxes):
    """This is for DIoU"""
    pred_ymin, pred_xmin, pred_ymax, pred_xmax = tf.unstack(pred_boxes, axis=1)
    gt_ymin, gt_xmin, gt_ymax, gt_xmax = tf.unstack(gt_boxes, axis=1)

    w = tf.minimum(pred_xmax, gt_xmax) - tf.maximum(pred_xmin, gt_xmin)
    h = tf.minimum(pred_ymax, gt_ymax) - tf.maximum(pred_ymin, gt_ymin)
    w = tf.clip_by_value(w, clip_value_min=0.0, clip_value_max=1.0)
    h = tf.clip_by_value(h, clip_value_min=0.0, clip_value_max=1.0)
    intersection = w * h
    w1 = pred_xmax - pred_xmin
    h1 = pred_ymax - pred_ymin
    w2 = gt_xmax - gt_xmin
    h2 = gt_ymax - gt_ymin
    union = (w1 * h1 + w2 * h2) - intersection
    EPSILON = 1e-14
    iou = intersection / (union + EPSILON)
    return iou


def compute_iou_and_union_pred_gt(pred_boxes, gt_boxes):
    """This is for DIoU"""
    pred_ymin, pred_xmin, pred_ymax, pred_xmax = tf.unstack(pred_boxes, axis=1)
    gt_ymin, gt_xmin, gt_ymax, gt_xmax = tf.unstack(gt_boxes, axis=1)

    w = tf.minimum(pred_xmax, gt_xmax) - tf.maximum(pred_xmin, gt_xmin)
    h = tf.minimum(pred_ymax, gt_ymax) - tf.maximum(pred_ymin, gt_ymin)
    w = tf.clip_by_value(w, clip_value_min=0.0, clip_value_max=1.0)
    h = tf.clip_by_value(h, clip_value_min=0.0, clip_value_max=1.0)
    intersection = w * h
    w1 = pred_xmax - pred_xmin
    h1 = pred_ymax - pred_ymin
    w2 = gt_xmax - gt_xmin
    h2 = gt_ymax - gt_ymin
    union = (w1 * h1 + w2 * h2) - intersection
    EPSILON = 1e-14
    iou = intersection / (union + EPSILON)
    return iou, union


def calc_p_square(pred_boxes, gt_boxes):
    """This is for DIoU"""
    pred_ycenter, pred_xcenter, pred_h, pred_w = to_center_coordinates(tf.unstack(pred_boxes, axis=1))
    gt_ycenter, gt_xcenter, gt_h, gt_w = to_center_coordinates(tf.unstack(gt_boxes, axis=1))
    p_square = (pred_xcenter - gt_xcenter) ** 2 + (pred_ycenter - gt_ycenter) ** 2
    return p_square


def calc_c_square(pred_boxes, gt_boxes):
    """This is for DIoU"""
    pred_ymin, pred_xmin, pred_ymax, pred_xmax = tf.unstack(pred_boxes, axis=1)
    gt_ymin, gt_xmin, gt_ymax, gt_xmax = tf.unstack(gt_boxes, axis=1)
    ymin = tf.minimum(pred_ymin, gt_ymin)
    xmin = tf.minimum(pred_xmin, gt_xmin)
    ymax = tf.maximum(pred_ymax, gt_ymax)
    xmax = tf.maximum(pred_xmax, gt_xmax)
    c_square = (xmax - xmin) ** 2 + (ymax - ymin) ** 2
    return c_square


def calc_c_area(pred_boxes, gt_boxes):
    """This is for DIoU"""
    pred_ymin, pred_xmin, pred_ymax, pred_xmax = tf.unstack(pred_boxes, axis=1)
    gt_ymin, gt_xmin, gt_ymax, gt_xmax = tf.unstack(gt_boxes, axis=1)
    ymin = tf.minimum(pred_ymin, gt_ymin)
    xmin = tf.minimum(pred_xmin, gt_xmin)
    ymax = tf.maximum(pred_ymax, gt_ymax)
    xmax = tf.maximum(pred_xmax, gt_xmax)
    c_area = (xmax - xmin) * (ymax - ymin)
    return c_area


def calc_landmark_loss(predictions, targets, weights, w=10.0, epsilon=2.0):
    """
    Arguments:
        predictions, targets: float tensors with shape [batch_size, num_anchor, 10].
        weights: float tensors with shape [batch, num_anchor, 10]
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    if params['use_landmark_wing_loss']:
        x = predictions - targets
        c = w * (1.0 - math.log(1.0 + w/epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(
            tf.greater(w, absolute_x),
            w * tf.log(1.0 + absolute_x/epsilon),
            absolute_x - c
        )

        losses = losses * weights
        #loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)
        loss = tf.reduce_sum(losses, axis=2)  # TODO: debug here
    else:
        abs_diff = tf.abs(predictions - targets)
        abs_diff_lt_1 = tf.less(abs_diff, 1.0)
        loss = weights * tf.reduce_sum(
            tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5), axis=2
        )
    return loss


def calc_classification_loss_v1(predictions, targets):
    """
    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, num_classes + 1],
            representing the predicted logits for each class.
        targets: an int tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    if params['use_class_label_smoothing']:
        #cross_entropy = tf.losses.softmax_cross_entropy(
        #    onehot_labels=targets,
        #    logits=predictions,
        #    label_smoothing=0.1  # [0, 1] ==> [0.05, 0.95]
        #)
        non_targets = 1 - targets
        targets = tf.stack([non_targets, targets], axis=-1)
        targets = tf.cast(targets, tf.float32)
        targets = \
            (1.0 - params['class_label_smoothing']) * targets + \
            0.5 * params['class_label_smoothing'] * tf.ones_like(targets)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=targets, logits=predictions
        )
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=predictions
        )
    return cross_entropy

def calc_classification_loss(predictions, targets):
    """
    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, num_classes + 1],
            representing the predicted logits for each class.
        targets: an int tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    if params['use_class_label_smoothing']:
        targets = \
            targets * (1 - params['class_label_smoothing']) + \
            0.5 * params['class_label_smoothing']
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=targets, logits=predictions
    )
    return cross_entropy


def calc_quality_loss(predictions, targets, weights):
    """
    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, num_classes + 1],
            representing the predicted logits for each class.
        targets: an int tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    #loss = tf.nn.sigmoid_cross_entropy_with_logits(
    #    labels=targets, logits=predictions
    #)

    #pos_weight = 10.0
    #loss = tf.nn.weighted_cross_entropy_with_logits(
    #    targets=targets, logits=predictions, pos_weight=pos_weight
    #)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=targets, logits=predictions)

    loss = loss * weights
    #loss = tf.squeeze(loss, axis=-1)
    return loss


def calc_label_loss(predictions, targets, weights):

    targets = tf.one_hot(indices=targets, depth=19, on_value=1, off_value=0, axis=-1)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=targets, logits=predictions)

    # loss = loss * weights
    # loss = tf.reduce_sum(loss)
    # loss = tf.

    return loss




def calc_blur_loss(predictions, targets, weights):
    """
    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, num_classes + 1],
            representing the predicted logits for each class.
        targets: an int tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=targets, logits=predictions
    # )

    pos_weight = 10.0
    loss = tf.nn.weighted_cross_entropy_with_logits(
        targets=targets, logits=predictions, pos_weight=pos_weight
    )
    loss = loss * weights
    return loss


def calc_occlude_loss(predictions, targets, weights):
    """
    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, num_classes + 1],
            representing the predicted logits for each class.
        targets: an int tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors,5].
    """
    if params['use_occlude_label_smoothing']:
        targets = \
            targets * (1 - params['occlude_label_smoothing']) + \
            0.5 * params['occlude_label_smoothing']
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=targets, logits=predictions
    )
    loss = loss * weights
    loss = tf.reduce_sum(loss, axis=2)

    return loss

def apply_hard_mining(
        location_losses,
        cls_losses,
        landmark_losses,
        label_losses,
        class_predictions_with_background,
        matches, decoded_boxes,
        loss_to_use='classification',
        loc_loss_weight=1.0, cls_loss_weight=1.0, lmk_loss_weight=1.0,
        num_hard_examples=3000, nms_threshold=0.99,
        max_negatives_per_positive=3, min_negatives_per_image=0):
    """Applies hard mining to anchorwise losses.

    Arguments:
        location_losses: a float tensor with shape [batch_size, num_anchors].
        cls_losses: a float tensor with shape [batch_size, num_anchors].
        class_predictions_with_background: a float tensor with shape [batch_size, num_anchors, num_classes + 1].
        matches: an int tensor with shape [batch_size, num_anchors].
        decoded_boxes: a float tensor with shape [batch_size, num_anchors, 4].
        loss_to_use: a string, only possible values are ['classification', 'both'].
        loc_loss_weight: a float number.
        cls_loss_weight: a float number.
        num_hard_examples: an integer.
        nms_threshold: a float number.
        max_negatives_per_positive: a float number.
        min_negatives_per_image: an integer.
    Returns:
        two float tensors with shape [].
    """

    # when training it is important that
    # batch size is known
    batch_size, num_anchors = matches.shape.as_list()
    assert batch_size is not None
    decoded_boxes.set_shape([batch_size, num_anchors, 4])
    location_losses.set_shape([batch_size, num_anchors])
    cls_losses.set_shape([batch_size, num_anchors])
    landmark_losses.set_shape([batch_size, num_anchors])
    label_losses.set_shape([batch_size, num_anchors])
    # quality_losses.set_shape([batch_size, num_anchors])
    # blur_losses.set_shape([batch_size, num_anchors])
    # occlude_losses.set_shape([batch_size, num_anchors])
    # all `set_shape` above are dirty tricks,
    # without them shape information is lost for some reason

    # all these tensors must have static first dimension (batch size)
    decoded_boxes_list = tf.unstack(decoded_boxes, axis=0)
    location_losses_list = tf.unstack(location_losses, axis=0)
    cls_losses_list = tf.unstack(cls_losses, axis=0)
    matches_list = tf.unstack(matches, axis=0)
    landmark_losses_list = tf.unstack(landmark_losses, axis=0)
    label_losses_list = tf.unstack(label_losses, axis=0)
    # quality_losses_list = tf.unstack(quality_losses, axis=0)
    # blur_losses_list = tf.unstack(blur_losses, axis=0)
    # occlude_losses_list = tf.unstack(occlude_losses, axis=0)
    # they all lists with length = batch_size

    batch_size = len(decoded_boxes_list)
    num_positives_list, num_negatives_list = [], []
    mined_location_losses = []
    mined_cls_losses = []
    mined_landmark_losses = []
    mined_label_losses = []
    # mined_quality_losses = []
    # mined_blur_losses = []
    # mined_occlude_losses = []

    # do OHEM for each image in the batch
    for i, box_locations in enumerate(decoded_boxes_list):
        image_losses = cls_losses_list[i] * cls_loss_weight
        if loss_to_use == 'both':
            image_losses += (location_losses_list[i] * loc_loss_weight)
        elif loss_to_use == 'three':
            image_losses += (location_losses_list[i] * loc_loss_weight)
            image_losses += (landmark_losses_list[i] * lmk_loss_weight)
        # it has shape [num_anchors]

        selected_indices = tf.image.non_max_suppression(
            box_locations, image_losses, num_hard_examples, nms_threshold
        )

        selected_indices, num_positives, num_negatives = \
            _subsample_selection_to_desired_neg_pos_ratio(
             selected_indices, matches_list[i],
             max_negatives_per_positive, min_negatives_per_image)

        num_positives_list.append(num_positives)
        num_negatives_list.append(num_negatives)
        mined_location_losses.append(
            tf.reduce_sum(tf.gather(location_losses_list[i], selected_indices), axis=0)
        )
        mined_landmark_losses.append(
            tf.reduce_sum(tf.gather(landmark_losses_list[i], selected_indices), axis=0)
        )
        mined_cls_losses.append(
            tf.reduce_sum(tf.gather(cls_losses_list[i], selected_indices), axis=0)
        )

        mined_label_losses.append(
            tf.reduce_sum(tf.gather(label_losses_list[i], selected_indices), axis=0)
        )
        '''
        mined_quality_losses.append(
            tf.reduce_sum(tf.gather(quality_losses_list[i], selected_indices), axis=0)
        )

        mined_blur_losses.append(
            tf.reduce_sum(tf.gather(blur_losses_list[i], selected_indices), axis=0)
        )

        mined_occlude_losses.append(
            tf.reduce_sum(tf.gather(occlude_losses_list[i], selected_indices), axis=0)
        )
        '''

    mean_num_positives = tf.reduce_mean(tf.stack(num_positives_list, axis=0), axis=0)
    mean_num_negatives = tf.reduce_mean(tf.stack(num_negatives_list, axis=0), axis=0)
    tf.summary.scalar('mean_num_positives', mean_num_positives)
    tf.summary.scalar('mean_num_negatives', mean_num_negatives)

    location_loss = tf.reduce_sum(tf.stack(mined_location_losses, axis=0), axis=0)
    landmark_loss = tf.reduce_sum(tf.stack(mined_landmark_losses, axis=0), axis=0)
    cls_loss = tf.reduce_sum(tf.stack(mined_cls_losses, axis=0), axis=0)
    label_loss = tf.reduce_sum(tf.stack(mined_label_losses, axis=0), axis=0)
    # quality_loss = tf.reduce_sum(tf.stack(mined_quality_losses, axis=0), axis=0)
    # blur_loss = tf.reduce_sum(tf.stack(mined_blur_losses, axis=0), axis=0)
    # occlude_loss = tf.reduce_sum(tf.stack(mined_occlude_losses, axis=0), axis=0)
    return location_loss, cls_loss, landmark_loss, label_loss


def _subsample_selection_to_desired_neg_pos_ratio(
        indices, match, max_negatives_per_positive, min_negatives_per_image):
    """Subsample a collection of selected indices to a desired neg:pos ratio.

    Arguments:
        indices: an int or long tensor with shape [M],
            it represents a collection of selected anchor indices.
        match: an int tensor with shape [num_anchors].
        max_negatives_per_positive: a float number, maximum number
            of negatives for each positive anchor.
        min_negatives_per_image: an integer, minimum number of negative anchors for a given
            image. Allows sampling negatives in image without any positive anchors.
    Returns:
        selected_indices: an int or long tensor with shape [M'] and with M' <= M.
            It represents a collection of selected anchor indices.
        num_positives: an int tensor with shape []. It represents the
            number of positive examples in selected set of indices.
        num_negatives: an int tensor with shape []. It represents the
            number of negative examples in selected set of indices.
    """
    positives_indicator = tf.gather(tf.greater_equal(match, 0), indices)
    negatives_indicator = tf.logical_not(positives_indicator)
    # they have shape [num_hard_examples]

    # all positives in `indices` will be kept
    num_positives = tf.reduce_sum(tf.to_int32(positives_indicator), axis=0)
    max_negatives = tf.maximum(
        min_negatives_per_image,
        tf.to_int32(max_negatives_per_positive * tf.to_float(num_positives))
    )

    top_k_negatives_indicator = tf.less_equal(
        tf.cumsum(tf.to_int32(negatives_indicator), axis=0),
        max_negatives
    )
    subsampled_selection_indices = tf.where(
        tf.logical_or(positives_indicator, top_k_negatives_indicator)
    )  # shape [num_hard_examples, 1]
    subsampled_selection_indices = tf.squeeze(subsampled_selection_indices, axis=1)
    selected_indices = tf.gather(indices, subsampled_selection_indices)

    num_negatives = tf.size(subsampled_selection_indices) - num_positives
    return selected_indices, num_positives, num_negatives
