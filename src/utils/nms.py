import tensorflow._api.v2.compat.v1 as tf
from src.constants import PARALLEL_ITERATIONS


def batch_non_max_suppression(
        boxes,
        landmarks,
        scores,
        labels,
        score_threshold,
        iou_threshold,
        max_boxes):
    """
    Arguments:
        boxes: a float tensor with shape [batch_size, N, 4].
        scores: a float tensor with shape [batch_size, N].
        score_threshold: a float number.
        iou_threshold: a float number, threshold for IoU.
        max_boxes: an integer, maximum number of retained boxes.
    Returns:
        boxes: a float tensor with shape [batch_size, max_boxes, 4].
        scores: a float tensor with shape [batch_size, max_boxes].
        num_detections: an int tensor with shape [batch_size].
    """
    def fn(x):
        boxes, scores, landmarks, labels = x

        # low scoring boxes are removed
        ids = tf.where(tf.greater_equal(scores, score_threshold))
        ids = tf.squeeze(ids, axis=1)
        boxes = tf.gather(boxes, ids)
        scores = tf.gather(scores, ids)
        landmarks = tf.gather(landmarks, ids)
        labels = tf.gather(labels, ids)
        # quality = tf.gather(quality, ids)
        # blur = tf.gather(blur, ids)
        # occlude = tf.gather(occlude, ids)

        selected_indices = tf.image.non_max_suppression(
            boxes, scores, max_boxes, iou_threshold
        )
        boxes = tf.gather(boxes, selected_indices)
        scores = tf.gather(scores, selected_indices)
        landmarks = tf.gather(landmarks, selected_indices)
        labels = tf.gather(labels, selected_indices)
        # quality = tf.gather(quality, selected_indices)
        # blur = tf.gather(blur, selected_indices)
        # occlude = tf.gather(occlude, selected_indices)

        num_boxes = tf.to_int32(tf.shape(boxes)[0])

        zero_padding = max_boxes - num_boxes
        boxes = tf.pad(boxes, [[0, zero_padding], [0, 0]])
        scores = tf.pad(scores, [[0, zero_padding]])
        landmarks = tf.pad(landmarks, [[0, zero_padding], [0, 0]])
        labels = tf.pad(labels, [[0, zero_padding], [0, 0]])
        # quality = tf.pad(quality, [[0, zero_padding]])
        # blur = tf.pad(blur, [[0, zero_padding]])
        # occlude = tf.pad(occlude, [[0, zero_padding], [0, 0]])

        boxes.set_shape([max_boxes, 4])
        scores.set_shape([max_boxes])
        landmarks.set_shape([max_boxes, 42])
        labels.set_shape([max_boxes, 19])
        # quality.set_shape([max_boxes])
        # blur.set_shape([max_boxes])
        # occlude.set_shape([max_boxes, 5])
        return boxes, scores, num_boxes, landmarks, labels

    boxes, scores, num_detections, landmarks, labels = tf.map_fn(
        fn, [boxes, scores, landmarks, labels],
        dtype=(tf.float32, tf.float32, tf.int32, tf.float32, tf.float32),
        parallel_iterations=PARALLEL_ITERATIONS,
        back_prop=False, swap_memory=False, infer_shape=True
    )
    return boxes, scores, num_detections, landmarks, labels


def batch_non_max_suppression_without_batch(
        boxes,
        landmarks,
        scores,
        labels,
        score_threshold,
        iou_threshold,
        max_boxes):

    #boxes, scores, landmarks, quality, blur, occlude

    # low scoring boxes are removed
    #ids = tf.where(tf.greater_equal(scores, score_threshold))
    #ids = tf.squeeze(ids, axis=1)
    #boxes = tf.gather(boxes, ids)
    #scores = tf.gather(scores, ids)
    #landmarks = tf.gather(landmarks, ids)
    #quality = tf.gather(quality, ids)
    #blur = tf.gather(blur, ids)
    #occlude = tf.gather(occlude, ids)


    selected_indices = tf.image.non_max_suppression(
        boxes, scores, max_boxes, iou_threshold
    )
    boxes = tf.gather(boxes, selected_indices)
    scores = tf.gather(scores, selected_indices)
    labels = tf.gather(labels, selected_indices)
    landmarks = tf.gather(landmarks, selected_indices)
    # quality = tf.gather(quality, selected_indices)
    # blur = tf.gather(blur, selected_indices)
    # occlude = tf.gather(occlude, selected_indices)

    num_boxes = tf.to_int32(tf.shape(boxes)[0])

    # zero_padding = max_boxes - num_boxes
    # boxes = tf.pad(boxes, [[0, zero_padding], [0, 0]])
    # scores = tf.pad(scores, [[0, zero_padding]])
    # labels = tf.pad(labels, [[0, zero_padding], [0, 0]])
    # landmarks = tf.pad(landmarks, [[0, zero_padding], [0, 0]])
    # quality = tf.pad(quality, [[0, zero_padding]])
    # blur = tf.pad(blur, [[0, zero_padding]])
    # occlude = tf.pad(occlude, [[0, zero_padding], [0, 0]])

    boxes.set_shape([max_boxes, 4])
    scores.set_shape([max_boxes])
    labels.set_shape([max_boxes, 19])
    landmarks.set_shape([max_boxes, 42])
    # quality.set_shape([max_boxes])
    # blur.set_shape([max_boxes])
    # occlude.set_shape([max_boxes, 5])
    return boxes, scores, num_boxes, landmarks, labels
