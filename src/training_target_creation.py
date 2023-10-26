import tensorflow._api.v2.compat.v1 as tf
from src.utils.box_utils import encode_boxes, encode_landmarks, iou
from src.utils.box_utils import L2_distance, get_top_k_mask, calc_iou_mean_std, choose_anchors_center_in_gt

from config import params
model_params = params['model_params']
input_params = params['input_pipeline_params']


def get_training_targets(
        anchors,
        num_anchors_per_feature_map,
        groundtruth_boxes,
        groundtruth_landmarks,
        groundtruth_landmark_weights,
        groundtruth_label,
        threshold=0.5):
    """
    Arguments:
        anchors: a float tensor with shape [num_anchors, 4].
        groundtruth_boxes: a float tensor with shape [N, 4].
        threshold: a float number.
    Returns:
        reg_targets: a float tensor with shape [num_anchors, 4].
        matches: an int tensor with shape [num_anchors], possible values
            that it can contain are [-1, 0, 1, 2, ..., (N - 1)].
    """
    with tf.name_scope('matching'):
        N = tf.shape(groundtruth_boxes)[0]
        num_anchors = tf.shape(anchors)[0]
        no_match_tensor = tf.fill([num_anchors], -1)
        matches = tf.cond(
            tf.greater(N, 0),
            #lambda: _match(anchors, num_anchors_per_feature_map, groundtruth_boxes, threshold),
            lambda: _match_ATSS(anchors, num_anchors_per_feature_map, groundtruth_boxes, threshold),
            lambda: no_match_tensor
        )
        matches = tf.to_int32(matches)

    with tf.name_scope('regression_target_creation'):
        reg_targets, \
        landmark_targets,\
        landmark_weights,\
        label_targets = \
            _create_targets(
                anchors,
                groundtruth_boxes,
                groundtruth_landmarks,
                groundtruth_landmark_weights,
                groundtruth_label,
                matches
            )

    return reg_targets, matches, landmark_targets, landmark_weights, label_targets


def _match_ATSS(anchors, num_anchors_per_feature_map, groundtruth_boxes, threshold=0.5):
    similarity_matrix = iou(
        groundtruth_boxes,
        anchors)  # shape [N, num_anchors]

    center_distance = L2_distance(groundtruth_boxes, anchors)

    top_k_mask = choose_top_k_on_feature_maps(center_distance, num_anchors_per_feature_map)  # shape [N, num_achors]
    sm = similarity_matrix * tf.to_float(top_k_mask)
    mean_iou, std_iou = calc_iou_mean_std(sm)
    thresh = mean_iou + std_iou  # shape [N]

    iou_mask = tf.greater_equal(tf.transpose(sm), thresh)
    iou_mask = tf.transpose(iou_mask)

    center_mask = choose_anchors_center_in_gt(groundtruth_boxes, anchors)
    mask = tf.to_int32(top_k_mask & iou_mask & center_mask)

    masked_similarity_matrix = similarity_matrix * tf.to_float(mask)

    matches = tf.argmax(
        masked_similarity_matrix,
        axis=0,
        output_type=tf.int32)  # shape [num_anchors]

    is_matched_anchor = tf.reduce_max(mask, axis=0)  # shape [num_anchors]

    matches = tf.add(
        tf.multiply(matches, is_matched_anchor),
        - (1 - is_matched_anchor))

    #num_anchors = tf.shape(anchors)[0]
    #forced_matches_ids = tf.argmax(
    #    masked_similarity_matrix,
    #    axis=1,
    #    output_type=tf.int32)  # shape [N]

    #forced_matches_indicators = tf.one_hot(
    #    forced_matches_ids,
    #    depth=num_anchors,
    #    dtype=tf.int32)  # shape [N, num_anchors]
    #forced_match_row_ids = tf.argmax(
    #    forced_matches_indicators,
    #    axis=0,
    #    output_type=tf.int32)  # shape [num_anchors]
    #forced_match_mask = tf.greater(
    #    tf.reduce_max(
    #        forced_matches_indicators,
    #        axis=0), 0)  # shape [num_anchors]
    #matches = tf.where(forced_match_mask, forced_match_row_ids, matches)

    return matches


def _match(anchors, num_anchors_per_feature_map, groundtruth_boxes, threshold=0.5):
    """Matching algorithm:
    1) for each groundtruth box choose the anchor with largest iou,
    2) remove this set of anchors from the set of all anchors,
    3) for each remaining anchor choose the groundtruth box with largest iou,
       but only if this iou is larger than `threshold`.

    Note: after step 1, it could happen that for some two groundtruth boxes
    chosen anchors are the same. Let's hope this never happens.
    Also see the comments below.

    Arguments:
        anchors: a float tensor with shape [num_anchors, 4].
        groundtruth_boxes: a float tensor with shape [N, 4].
        threshold: a float number.
    Returns:
        an int tensor with shape [num_anchors].
    """
    num_anchors = tf.shape(anchors)[0]

    # for each anchor box choose the groundtruth box with largest iou
    similarity_matrix = iou(
        groundtruth_boxes,
        anchors)  # shape [N, num_anchors]

    #similarity_matrix = \
    #    limit_target_size_on_feature_maps(
    #        similarity_matrix, num_anchors_per_feature_map, groundtruth_boxes)

    #center_distance = L2_distance(groundtruth_boxes, anchors)
    #mask = choose_top_k_on_feature_maps(center_distance, num_anchors_per_feature_map)
    #mean_iou, std_iou = calc_iou_mean_std(similarity_matrix * mask)

    matches = tf.argmax(
        similarity_matrix,
        axis=0,
        output_type=tf.int32)  # shape [num_anchors]
    matched_vals = tf.reduce_max(
        similarity_matrix,
        axis=0)  # shape [num_anchors]
    below_threshold = tf.to_int32(tf.greater(threshold, matched_vals))
    matches = tf.add(
        tf.multiply(matches, 1 - below_threshold),  # get anchor matched gt idx above threshold
        -1 * below_threshold)  # below_threshold would be -1

    # after this, it could happen that some groundtruth
    # boxes are not matched with any anchor box

    # now we must ensure that each row (groundtruth box) is matched to
    # at least one column (which is not guaranteed
    # otherwise if `threshold` is high)

    # for each groundtruth box choose the anchor box with largest iou
    # (force match for each groundtruth box)
    forced_matches_ids = tf.argmax(
        similarity_matrix,
        axis=1,
        output_type=tf.int32)  # shape [N]
    # if all indices in forced_matches_ids are different then all rows will be
    # matched

    forced_matches_indicators = tf.one_hot(
        forced_matches_ids,
        depth=num_anchors,
        dtype=tf.int32)  # shape [N, num_anchors]
    forced_match_row_ids = tf.argmax(
        forced_matches_indicators,
        axis=0,
        output_type=tf.int32)  # shape [num_anchors]
    forced_match_mask = tf.greater(
        tf.reduce_max(
            forced_matches_indicators,
            axis=0), 0)  # shape [num_anchors]
    matches = tf.where(forced_match_mask, forced_match_row_ids, matches)
    # even after this it could happen that some rows aren't matched,
    # but i believe that this event has low probability

    return matches


def limit_target_size_on_feature_maps(
        similarity_matrix,
        num_anchors_per_feature_map,
        groundtruth_boxes):
    """
    for each feature map limit the ground truth size to corresponding anchors scale size
    :param similarity_matrix:  shape [N, num_anchors]
    :param num_anchors_per_feature_map: a list shape [n features]
    :param groundtruth_boxes:  shape [N, 4] ymin, xmin, ymax, xmax
    :return: similarity_matrix
    """
    sm_splits = tf.split(similarity_matrix, num_anchors_per_feature_map, axis=1)
    ymin, xmin, ymax, xmax = tf.split(groundtruth_boxes, num_or_size_splits=4, axis=1)
    w = xmax - xmin
    h = ymax - ymin
    img_h, img_w = input_params['image_size']
    assert img_w == img_h

    '''
    range_list = [
        (5, 15),
        (15, 20),
        (20, 40),
        (40, 80),
        (80, 150),
        (150, 320),
        (320, 512)
    ]
    '''
    range_list = [
        (5, 24),
        (24, 48),
        (48, 96),
        (96, 512),
    ]
    assert len(range_list) == len(num_anchors_per_feature_map)

    long_side = tf.maximum(w, h) * img_h  # shape [N]
    sm_list = []
    for i, sm_one_feature in enumerate(sm_splits):
        range_min, range_max = range_list[i]
        num_anchors = num_anchors_per_feature_map[i]
        mask = tf.greater_equal(long_side, range_min)
        mask &= tf.less(long_side, range_max)
        #mask = tf.expand_dims(mask, axis=1)
        mask = tf.tile(mask, [1, num_anchors])  # make it [N, num_anchors]
        sm_list.append(sm_one_feature * tf.to_float(mask))

    sm = tf.concat(sm_list, axis=1)

    return sm


def choose_top_k_on_feature_maps(center_distance, num_anchors_per_feature_map):
    cd_splits = tf.split(center_distance, num_anchors_per_feature_map, axis=1)
    mask_list = []
    for i, cd_one_feature in enumerate(cd_splits):
        mask = get_top_k_mask(cd_one_feature, k=9)
        mask_list.append(mask)
    all_anchors_mask = tf.concat(mask_list, axis=1)
    return all_anchors_mask


def _create_targets(
        anchors,
        groundtruth_boxes,
        groundtruth_landmarks,
        groundtruth_landmark_weights,
        groundtruth_label,
        matches):
    """Returns regression targets for each anchor.

    Arguments:
        anchors: a float tensor with shape [num_anchors, 4].
        groundtruth_boxes: a float tensor with shape [N, 4].
        matches: a int tensor with shape [num_anchors].
    Returns:
        reg_targets: a float tensor with shape [num_anchors, 4].
    """
    matched_anchor_indices = tf.where(tf.greater_equal(matches, 0))  # shape [num_matches, 1]
    matched_anchor_indices = tf.squeeze(matched_anchor_indices, axis=1)
    matched_gt_indices = tf.gather(matches, matched_anchor_indices)  # shape [num_matches]

    matched_anchors = tf.gather(anchors, matched_anchor_indices)  # shape [num_matches, 4]
    matched_gt_boxes = tf.gather(groundtruth_boxes, matched_gt_indices)  # shape [num_matches, 4]
    if model_params['use_diou_loss'] == False:
        matched_reg_targets = encode_boxes(matched_gt_boxes, matched_anchors)  # shape [num_matches, 4]
    else:
        matched_reg_targets = matched_gt_boxes  # shape [num_matches, 4]

    matched_gt_landmarks = tf.gather(groundtruth_landmarks, matched_gt_indices)  # shape [num_matches, 10]
    matched_landmark_targets = encode_landmarks(matched_gt_landmarks, matched_anchors)  # shape [num_matches, 10]
    matched_gt_landmark_weights = tf.gather(groundtruth_landmark_weights, matched_gt_indices)  # shape [num_matches, 10]
    matched_gt_label_targets = tf.gather(groundtruth_label, matched_gt_indices)
    # matched_gt_quality_targets = tf.gather(groundtruth_quality, matched_gt_indices)  # shape [num_matches, 1]
    # matched_gt_blur_targets = tf.gather(groundtruth_blur, matched_gt_indices)  # shape [num_matches, 1]
    # matched_gt_occlude_targets = tf.gather(groundtruth_occlude, matched_gt_indices)  # shape [num_matches, 5]

    unmatched_anchor_indices = tf.where(tf.equal(matches, -1))
    unmatched_anchor_indices = tf.squeeze(unmatched_anchor_indices, axis=1)
    # it has shape [num_anchors - num_matches]

    unmatched_reg_targets = tf.zeros([tf.size(unmatched_anchor_indices), 4])
    unmatched_landmark_targets = tf.zeros([tf.size(unmatched_anchor_indices), 42])
    unmatched_label_targets = tf.zeros([tf.size(unmatched_anchor_indices)], dtype=tf.int32)
    # unmatched_quality_targets = tf.zeros([tf.size(unmatched_anchor_indices)])
    # unmatched_blur_targets = tf.zeros([tf.size(unmatched_anchor_indices)])
    # unmatched_occlude_targets = tf.zeros([tf.size(unmatched_anchor_indices), 5])
    # it has shape [num_anchors - num_matches, 4]

    matched_anchor_indices = tf.to_int32(matched_anchor_indices)
    unmatched_anchor_indices = tf.to_int32(unmatched_anchor_indices)

    reg_targets = tf.dynamic_stitch(
        [matched_anchor_indices, unmatched_anchor_indices],
        [matched_reg_targets, unmatched_reg_targets]
    )

    landmark_targets = tf.dynamic_stitch(
        [matched_anchor_indices, unmatched_anchor_indices],
        [matched_landmark_targets, unmatched_landmark_targets]
    )

    landmark_weights = tf.dynamic_stitch(
        [matched_anchor_indices, unmatched_anchor_indices],
        [matched_gt_landmark_weights, unmatched_landmark_targets]
    )

    label_targets = tf.dynamic_stitch(
        [matched_anchor_indices, unmatched_anchor_indices],
        [matched_gt_label_targets, unmatched_label_targets]
    )
    '''

    quality_targets = tf.dynamic_stitch(
        [matched_anchor_indices, unmatched_anchor_indices],
        [matched_gt_quality_targets, unmatched_quality_targets]
    )

    blur_targets = tf.dynamic_stitch(
        [matched_anchor_indices, unmatched_anchor_indices],
        [matched_gt_blur_targets, unmatched_blur_targets]
    )

    occlude_targets = tf.dynamic_stitch(
        [matched_anchor_indices, unmatched_anchor_indices],
        [matched_gt_occlude_targets, unmatched_occlude_targets]
    )
    '''

    return reg_targets, landmark_targets, landmark_weights, label_targets
