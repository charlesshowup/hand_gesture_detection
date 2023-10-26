import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import tf_slim as slim


import math
from config import params

from src.constants import MATCHING_THRESHOLD, PARALLEL_ITERATIONS, BATCH_NORM_MOMENTUM, RESIZE_METHOD
from src.utils.box_utils import batch_decode
from src.utils.nms import batch_non_max_suppression, batch_non_max_suppression_without_batch
from src.training_target_creation import get_training_targets
from src.losses_and_ohem import \
    calc_localization_loss, \
    calc_localization_diou_loss, \
    calc_classification_loss, \
    calc_landmark_loss, \
    calc_quality_loss, \
    calc_blur_loss, \
    calc_occlude_loss,\
    calc_label_loss, \
    apply_hard_mining

from backbone import *
#from src.lffdv1 import *
#from src.face_net_org import *
#from src.net import *
#from src.vovnet_face_test import *
#from src.vovnet_face_1 import *
# if params['quantization_params']['use_quantization_model']:
#     from src.model.net_inference_W8bitA4bit import *
# else:
#     from src.model.net_inference_float import *


class Detector:
    def __init__(self, images, feature_extractor, anchor_generator):
        """
        Arguments:
            images: a float tensor with shape [batch_size, height, width, 3],
                a batch of RGB images with pixel values in the range [0, 1].
            feature_extractor: an instance of FeatureExtractor.
            anchor_generator: an instance of AnchorGenerator.
        """

        org_h, org_w = tf.shape(images)[1], tf.shape(images)[2]
        total_stride = feature_extractor.get_total_stride()
        images = self.pad_images(images, total_stride)
        padded_h, padded_w = tf.shape(images)[1], tf.shape(images)[2]

        self.set_coord_scale(org_h, padded_h, org_w, padded_w)

        feature_maps = feature_extractor.extract_feat(images)
        self.is_training = feature_extractor.is_training

        self.anchors = anchor_generator(feature_maps, image_size=(padded_w, padded_h))
        self.num_anchors_per_location = anchor_generator.num_anchors_per_location
        self.num_anchors_per_feature_map = anchor_generator.num_anchors_per_feature_map
        self.anchor_grid_list = anchor_generator.anchor_grid_list
        self._add_box_predictions(feature_maps)

    def set_coord_scale(self, h, new_h, w, new_w):
        with tf.name_scope('set_coord_scale'):
            self.box_scaler = tf.to_float(tf.stack([
                h / new_h, w / new_w, h / new_h, w / new_w
            ]))

            self.landmark_scaler = tf.to_float(tf.stack([
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,
                w / new_w, h / new_h,



            ]))


    def pad_images(self, images, total_stride):
        """
         image padding here is very tricky and important part of the detector,
         if we don't do it then some bounding box
         predictions will be badly shifted!
        :param images:
        :param total_stride: the last layer stride
        :return:
        """
        x = total_stride
        h, w = images.shape.as_list()[1:3]  # do not write _, h, w, _ = images.shape, it would get dimension
        if h is None or w is None or h % x != 0 or w % x != 0:
            h, w = tf.shape(images)[1], tf.shape(images)[2]
            with tf.name_scope('image_padding'):
                # image size must be divisible by x
                new_h = x * tf.to_int32(tf.ceil(h / x))
                new_w = x * tf.to_int32(tf.ceil(w / x))
                images = tf.image.pad_to_bounding_box(
                    images, offset_height=0, offset_width=0,
                    target_height=new_h, target_width=new_w
                )
        return images

    def get_predictions(self, score_threshold=0.1,
                        iou_threshold=0.6, max_boxes=20):
        """Postprocess outputs of the network.

        Returns:
            boxes: a float tensor with shape [batch_size, N, 4].
            scores: a float tensor with shape [batch_size, N].
            num_boxes: an int tensor with shape [batch_size], it
                represents the number of detections on an image.

            where N = max_boxes.
        """
        with tf.name_scope('postprocessing'):
            boxes, landmarks = batch_decode(
                self.box_encodings, self.anchors, self.landmark_encodings)
            # if the images were padded we need to rescale predicted boxes:
            boxes = boxes / self.box_scaler
            boxes = tf.clip_by_value(boxes, 0.0, 1.0)
            # it has shape [batch_size, num_anchors, 4]

            landmarks = landmarks / self.landmark_scaler
            landmarks = tf.clip_by_value(landmarks, 0.0, 1.0)


            #scores = tf.nn.softmax(
            #    self.class_predictions_with_background, axis=2)[
            #    :, :, 1]
            scores = tf.nn.sigmoid(self.class_predictions_with_background)
            # it has shape [batch_size, num_anchors]

            #quality = tf.nn.sigmoid(self.quality_logits, axis=2)[:, :, 0]
            #quality = tf.nn.sigmoid(self.quality_logits)[:, :, 0]
            labels = tf.nn.softmax(self.label_logits)
            # quality = tf.nn.sigmoid(self.quality_logits)
            # blur = tf.nn.sigmoid(self.blur_logits)
            # occlude = tf.nn.sigmoid(self.occlude_logits)


        with tf.device('/cpu:0'), tf.name_scope('nms'):
            boxes, scores, num_detections, landmarks, labels = \
                batch_non_max_suppression(
                    boxes,
                    landmarks,
                    scores,
                    labels,
                    score_threshold,
                    iou_threshold,
                    max_boxes
                )
        '''
        with tf.device('/cpu:0'), tf.name_scope('nms'):
            boxes, scores, num_detections, landmarks, labels = \
                batch_non_max_suppression_without_batch(
                    boxes[0],
                    landmarks[0],
                    scores[0],
                    labels[0],
                    score_threshold,
                    iou_threshold,
                    max_boxes
                )

        # with tf.device('/cpu:0'), tf.name_scope('nms'):
        #     boxes, scores, num_detections, labels = \
        #         batch_non_max_suppression_without_batch(
        #             boxes[0],
        #             scores[0],
        #             labels[0],
        #             score_threshold,
        #             iou_threshold,
        #             max_boxes
        #         )
        '''
        return {'boxes': boxes,
                'scores': scores,
                'num_boxes': num_detections,
                'landmarks': landmarks,
                'gesture_labels': labels
                }

    def loss(self, groundtruth, params):
        """Compute scalar loss tensors with respect to provided groundtruth.

        Arguments:
            groundtruth: a dict with the following keys
                'boxes': a float tensor with shape [batch_size, max_num_boxes, 4].
                'num_boxes': an int tensor with shape [batch_size].
                    where max_num_boxes = max(num_boxes).
            params: a dict with parameters for OHEM.
        Returns:
            two float tensors with shape [].
        """

        max_num_boxes = tf.reduce_max(groundtruth['num_boxes'])
        gt_occlude = tf.zeros_like(groundtruth['landmarks'])[:, :max_num_boxes, :]
        # gt_occlude = groundtruth['landmark_occlude'][:, :max_num_boxes, :]
        landmark_gt_weights = self.trans_gt_occlude_to_gt_weights(gt_occlude, max_num_boxes)

        reg_targets, \
        matches, \
        landmark_targets, \
        landmark_weights, \
        label_targets = \
            self._create_targets(groundtruth, landmark_gt_weights)

        with tf.name_scope('losses'):

            # whether anchor is matched
            is_matched = tf.greater_equal(matches, 0)
            weights = tf.to_float(is_matched)
            # shape [batch_size, num_anchors]

            # we have binary classification for each anchor
            cls_targets = tf.to_float(is_matched)

            with tf.name_scope('classification_loss'):
                cls_losses = calc_classification_loss(
                    self.class_predictions_with_background,
                    cls_targets
                )

            with tf.name_scope('localization_loss'):
                if params['use_diou_loss']:
                    location_losses = calc_localization_diou_loss(
                        self.box_encodings,
                        reg_targets,
                        self.anchors,
                        weights
                    )
                else:
                    location_losses = calc_localization_loss(
                        self.box_encodings,
                        reg_targets, weights
                    )

                total_loss = tf.reduce_sum(location_losses)
                loc_loss_splits = \
                    tf.split(location_losses, self.num_anchors_per_feature_map, axis=1)
                for i, loc_loss in enumerate(loc_loss_splits):
                    tf.summary.scalar(
                        'fm%d_loss_ratio' % i,
                        tf.reduce_sum(loc_loss) / total_loss
                    )

            with tf.name_scope('landmark_loss'):
                landmark_losses = calc_landmark_loss(
                    self.landmark_encodings,
                    landmark_targets, landmark_weights
                )


            with tf.name_scope('label_loss'):
                label_weights = tf.reduce_sum(landmark_weights, axis=-1)
                label_weights = \
                    tf.where(
                        tf.equal(label_weights, 0.0),
                        tf.zeros_like(label_weights),
                        tf.ones_like(label_weights)
                    )
                # label_weights = label_weights[:, :, 0::2]
                label_losses = calc_label_loss(self.label_logits,
                                               label_targets,
                                               label_weights)

            '''

            with tf.name_scope('quality_loss'):
                quality_weights = tf.reduce_sum(landmark_weights, axis=-1)
                quality_weights = \
                    tf.where(
                        tf.equal(quality_weights, 0.0),
                        tf.zeros_like(quality_weights),
                        tf.ones_like(quality_weights))
                quality_losses = calc_quality_loss(
                    self.quality_logits,
                    quality_targets,
                    quality_weights
                )

            with tf.name_scope('blur_loss'):
                blur_losses = calc_blur_loss(
                    self.blur_logits,
                    blur_targets,
                    weights
                )

            with tf.name_scope('occlude_loss'):
                occlude_weights = \
                    tf.where(
                        tf.equal(landmark_weights, 0.5),
                        tf.ones_like(landmark_weights),
                        landmark_weights)
                occlude_weights = occlude_weights[:, :, 0::2]

                occlude_losses = calc_occlude_loss(
                    self.occlude_logits,
                    occlude_targets,
                    occlude_weights
                )
            # they have shape [batch_size, num_anchors]
            '''
            with tf.name_scope('normalization'):
                matches_per_image = \
                    tf.reduce_sum(weights, axis=1)  # shape [batch_size]
                num_matches = tf.reduce_sum(matches_per_image)  # shape []
                normalizer = tf.maximum(num_matches, 1.0)

                matches_per_image = \
                    tf.reduce_sum(landmark_weights, axis=[1, 2])  # shape [batch_size]
                num_matches = tf.reduce_sum(matches_per_image)  # shape []
                landmark_normalizer = tf.maximum(num_matches, 1.0)
                '''

                matches_per_image = \
                    tf.reduce_sum(occlude_weights, axis=[1, 2])  # shape [batch_size]
                num_matches = tf.reduce_sum(matches_per_image)  # shape []
                occlude_normalizer = tf.maximum(num_matches, 1.0)
                '''

            #scores = tf.nn.softmax(
            #    self.class_predictions_with_background, axis=2)

            # it has shape [batch_size, num_anchors, 2]

            decoded_boxes, decoded_landmarks = batch_decode(
                self.box_encodings, self.anchors, self.landmark_encodings)
            # it has shape [batch_size, num_anchors, 4]
            decoded_boxes = decoded_boxes / self.box_scaler
            #decoded_landmarks = decoded_landmarks / self.landmark_scaler

            if 0:
                # add summaries for predictions
                is_background = tf.equal(matches, -1)
                self._add_scalewise_histograms(tf.to_float(
                    is_background) * scores[:, :, 0], 'background_probability')
                self._add_scalewise_histograms(
                    weights * scores[:, :, 1], 'face_probability')
                ymin, xmin, ymax, xmax = tf.unstack(decoded_boxes, axis=2)
                h, w = ymax - ymin, xmax - xmin
                self._add_scalewise_histograms(weights * h, 'box_heights')
                self._add_scalewise_histograms(weights * w, 'box_widths')

                # add summaries for losses and matches
                self._add_scalewise_matches_summaries(weights)
                self._add_scalewise_summaries(
                    cls_losses, name='classification_losses')
                self._add_scalewise_summaries(
                    location_losses, name='localization_losses')
                self._add_scalewise_summaries(
                    landmark_losses, name='landmark_losses')
                self._add_scalewise_summaries(
                    quality_losses, name='quality_losses')
                self._add_scalewise_summaries(
                    blur_losses, name='blur_losses')
                self._add_scalewise_summaries(
                    occlude_losses, name='occlude_losses')
                tf.summary.scalar(
                    'total_mean_matches_per_image',
                    tf.reduce_mean(matches_per_image))

            with tf.name_scope('ohem'):
                location_loss, \
                cls_loss, \
                landmark_loss,\
                label_loss  = apply_hard_mining(
                    location_losses,
                    cls_losses,
                    landmark_losses,
                    label_losses,
                    self.class_predictions_with_background,
                    matches, decoded_boxes,
                    loss_to_use=params['loss_to_use'],
                    loc_loss_weight=params['loc_loss_weight'],
                    cls_loss_weight=params['cls_loss_weight'],
                    lmk_loss_weight=params['lmk_loss_weight'],
                    num_hard_examples=params['num_hard_examples'],
                    nms_threshold=params['nms_threshold'],
                    max_negatives_per_positive=params['max_negatives_per_positive'],
                    min_negatives_per_image=params['min_negatives_per_image']
                )
                return {'localization_loss': location_loss / normalizer,
                        'landmark_loss': landmark_loss / landmark_normalizer,
                        'classification_loss': cls_loss / normalizer,
                        'label_loss': label_loss / normalizer,
                        # 'quality_loss': quality_loss / normalizer,
                        # 'blur_loss': blur_loss / normalizer,
                        # 'occlude_loss': occlude_loss / occlude_normalizer,
                        }

    def trans_gt_occlude_to_gt_weights(self, gt_occlude, max_num_boxes):
        # TODO: debug here
        """
        :param gt_weights: [batch_size, max_num_boxes, 5]  occlude: 0.0, 1.0 -1.0
        :return gt_weights: [batch_size, max_num_boxes, 10] weight: 1.0, 0.5 0.0
        """
        gt_weights = \
            tf.where(tf.equal(gt_occlude, 1.0), 0.5*tf.ones_like(gt_occlude), gt_occlude)
        gt_weights = \
            tf.where(tf.equal(gt_weights, 0.0), tf.ones_like(gt_weights), gt_weights)
        gt_weights = \
            tf.where(tf.equal(gt_weights, -1.0), tf.zeros_like(gt_weights), gt_weights)

        gt_weights = tf.expand_dims(gt_weights, axis=-1)
        gt_weights = tf.tile(gt_weights, [1, 1, 1, 1])

        batch_size, _, _, _ = gt_weights.shape
        gt_weights = tf.reshape(gt_weights, [batch_size, max_num_boxes, 42])

        return gt_weights

    def _add_scalewise_summaries(self, tensor, name, percent=0.2):
        """Adds histograms of the biggest 20 percent of
        tensor's values for each scale (feature map).

        Arguments:
            tensor: a float tensor with shape [batch_size, num_anchors].
            name: a string.
            percent: a float number, default value is 20%.
        """
        index = 0
        for i, n in enumerate(self.num_anchors_per_feature_map):
            k = tf.ceil(tf.to_float(n) * percent)
            k = tf.to_int32(k)
            biggest_values, _ = tf.nn.top_k(
                tensor[:, index:(index + n)], k, sorted=False)
            # it has shape [batch_size, k]
            tf.summary.histogram(
                name + '_on_scale_' + str(i),
                tf.reduce_mean(biggest_values, axis=0)
            )
            index += n

    def _add_scalewise_histograms(self, tensor, name):
        """Adds histograms of the tensor's nonzero values for each scale (feature map).

        Arguments:
            tensor: a float tensor with shape [batch_size, num_anchors].
            name: a string.
        """
        index = 0
        for i, n in enumerate(self.num_anchors_per_feature_map):
            values = tf.reshape(tensor[:, index:(index + n)], [-1])
            nonzero = tf.greater(values, 0.0)
            values = tf.boolean_mask(values, nonzero)
            tf.summary.histogram(name + '_on_scale_' + str(i), values)
            index += n

    def _add_scalewise_matches_summaries(self, weights):
        """Adds summaries for the number of matches on each scale."""
        index = 0
        for i, n in enumerate(self.num_anchors_per_feature_map):
            matches_per_image = tf.reduce_sum(
                weights[:, index:(index + n)], axis=1)
            tf.summary.scalar(
                'mean_matches_per_image_on_scale_' + str(i),
                tf.reduce_mean(matches_per_image, axis=0)
            )
            index += n

    def _create_targets(self, groundtruth, landmark_gt_weights):
        """
        Arguments:
            groundtruth: a dict with the following keys
                'boxes': a float tensor with shape [batch_size, N, 4].
                'num_boxes': an int tensor with shape [batch_size].
        Returns:
            reg_targets: a float tensor with shape [batch_size, num_anchors, 4].
            matches: an int tensor with shape [batch_size, num_anchors].
        """
        def fn(x):
            boxes, num_boxes, landmarks, landmark_gt_weights, labels = x

            boxes = boxes[:num_boxes]
            boxes = boxes * self.box_scaler

            landmarks = landmarks[:num_boxes]
            landmarks = landmarks * self.landmark_scaler

            landmark_gt_weights = landmark_gt_weights[:num_boxes]

            labels = labels[:num_boxes]
            # labels = tf.cast(labels, dtype=tf.int32)
            '''
            quality = quality[:num_boxes]

            blur = blur[:num_boxes]
            occlude = occlude[:num_boxes]
            '''

            # if the images are padded we need to rescale groundtruth boxes:
            reg_targets, \
            matches, \
            landmark_targets, \
            landmark_weights, \
            label_targets = \
                get_training_targets(
                    self.anchors,
                    self.num_anchors_per_feature_map,
                    boxes,
                    landmarks,
                    landmark_gt_weights,
                    labels,
                    threshold=MATCHING_THRESHOLD
                )

            return reg_targets, matches, landmark_targets, landmark_weights, label_targets

        with tf.name_scope('target_creation'):
            reg_targets, \
            matches, \
            landmark_targets, \
            landmark_weights, \
            label_targets = tf.map_fn(
                fn,
                [groundtruth['boxes'],
                 groundtruth['num_boxes'],
                 groundtruth['landmarks'],
                 landmark_gt_weights,
                 groundtruth['gesture_labels']
                 ],
                dtype=(tf.float32, tf.int32, tf.float32, tf.float32, tf.int32),
                parallel_iterations=PARALLEL_ITERATIONS,
                back_prop=False, swap_memory=False, infer_shape=True
            )
            return reg_targets, matches, landmark_targets, landmark_weights, label_targets
    def _add_box_predictions(self, feature_maps):
        """Adds box predictors to each feature map, reshapes, and returns concatenated results.

        Arguments:
            feature_maps: a list of float tensors where the ith tensor has shape
                [batch, height_i, width_i, channels_i].

        It creates two tensors:
            box_encodings: a float tensor with shape [batch_size, num_anchors, 4].
            class_predictions_with_background: a float tensor with shape
                [batch_size, num_anchors, 2].
        """
        num_anchors_per_location = self.num_anchors_per_location
        num_feature_maps = len(feature_maps)
        box_encodings = []
        landmark_encodings = []
        class_predictions_with_background = []
        label_logits = []
        # quality_logits = []
        # blur_logits = []
        # occlude_logits = []

        if params['model_params']['is_fine_tune_landmark']:
            trainable = False
            landmark_trainable = True
        else:
            trainable = True
            landmark_trainable = True

        def batch_norm(x):
            x = tf.layers.batch_normalization(
                x, axis=3, center=True, scale=True,
                momentum=BATCH_NORM_MOMENTUM, epsilon=0.001,
                training=trainable,
                trainable=trainable,
                fused=True,
                name='batch_norm'
            )
            return x

        with tf.variable_scope('prediction_layers'):
            for i in range(num_feature_maps):
                # box
                feature = feature_maps[i]
                num_predictions_per_location = num_anchors_per_location[i]

                y = create_box_head(feature, i, num_predictions_per_location, trainable)
                box_encodings.append(y)

                # landmark
                y = create_landmark_head(feature, i, num_predictions_per_location, landmark_trainable)
                landmark_encodings.append(y)

                # class
                y = create_class_head(feature, i, num_predictions_per_location, trainable)
                class_predictions_with_background.append(y)

                #label
                y = create_label_head(feature, i, num_predictions_per_location, trainable)
                label_logits.append(y)
                '''

                # quality
                y = create_quality_head(feature, i, num_predictions_per_location, trainable)
                quality_logits.append(y)

                # blur
                y = create_blur_head(feature, i, num_predictions_per_location, trainable)
                blur_logits.append(y)

                # occlude
                y = create_occlude_head(feature, i, num_predictions_per_location, trainable)
                occlude_logits.append(y)
                '''


        # it is important that reshaping here is the same as when anchors were
        # generated
        with tf.name_scope('reshaping'):
            for i in range(num_feature_maps):
                try:
                    tensor_size = tf.shape(feature_maps[i])
                except:
                    tensor_size = tf.shape(feature_maps[i].outputs)

                num_predictions_per_location = num_anchors_per_location[i]
                batch_size = tensor_size[0]
                height_i = tensor_size[1]
                width_i = tensor_size[2]
                num_anchors_on_feature_map = height_i * width_i * num_predictions_per_location

                y = box_encodings[i]
                y = tf.reshape(y, tf.stack(
                    [batch_size, height_i, width_i, num_predictions_per_location, 4]))
                box_encodings[i] = tf.reshape(
                    y, [batch_size, num_anchors_on_feature_map, 4])

                y = landmark_encodings[i]
                y = tf.reshape(y, tf.stack(
                    [batch_size, height_i, width_i, num_predictions_per_location, 42]))
                landmark_encodings[i] = tf.reshape(
                    y, [batch_size, num_anchors_on_feature_map, 42])

                y = class_predictions_with_background[i]
                y = tf.reshape(
                    y, [batch_size, height_i, width_i, num_predictions_per_location])
                class_predictions_with_background[i] = tf.reshape(
                    y, tf.stack([batch_size, num_anchors_on_feature_map]))

                y = label_logits[i]
                y = tf.reshape(
                    y, [batch_size, height_i, width_i, num_predictions_per_location, 19])
                label_logits[i] = tf.reshape(
                    y, tf.stack([batch_size, num_anchors_on_feature_map, 19]))

                '''
                y = quality_logits[i]
                y = tf.reshape(
                    y, [batch_size, height_i, width_i, num_predictions_per_location])
                quality_logits[i] = tf.reshape(
                    y, tf.stack([batch_size, num_anchors_on_feature_map]))

                y = blur_logits[i]
                y = tf.reshape(
                    y, [batch_size, height_i, width_i, num_predictions_per_location])
                blur_logits[i] = tf.reshape(
                    y, tf.stack([batch_size, num_anchors_on_feature_map]))

                y = occlude_logits[i]
                y = tf.reshape(y, tf.stack(
                    [batch_size, height_i, width_i, num_predictions_per_location, 5]))
                occlude_logits[i] = tf.reshape(
                    y, [batch_size, num_anchors_on_feature_map, 5])
                '''
            self.box_encodings = tf.concat(box_encodings, axis=1)
            self.landmark_encodings = tf.concat(landmark_encodings, axis=1)
            self.class_predictions_with_background = tf.concat(
                class_predictions_with_background, axis=1)
            self.label_logits = tf.concat(label_logits, axis=1)
            # self.quality_logits = tf.concat(quality_logits, axis=1)
            # self.blur_logits = tf.concat(blur_logits, axis=1)
            # self.occlude_logits = tf.concat(occlude_logits, axis=1)


def make_landmark_head(x, c, trainable, i):
    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=3, center=True, scale=True,
            momentum=BATCH_NORM_MOMENTUM, epsilon=0.001,
            training=trainable,
            trainable=trainable,
            fused=True,
            name='batch_norm'
        )
        return x
    c = int(c)
    params = {
        'padding': 'SAME',
        'activation_fn': tf.nn.relu,
        'normalizer_fn': batch_norm, 'data_format': 'NHWC',
        'trainable': trainable
    }
    with slim.arg_scope([slim.conv2d], **params):
        #if i == 0:
        #    x = slim.conv2d(x, c / 4, [1, 1],
        #                    scope='landmark_enc_pred_%d_blk_0_conv1x1' % i)
        #elif i == 1:
        #    x = slim.conv2d(x, c / 4, [1, 1],
        #                    scope='landmark_enc_pred_%d_blk_0_conv1x1' % i)
        #    x = slim.conv2d(x, c / 2, [3, 3],
        #                    scope='landmark_enc_pred_%d_blk_0_conv3x3' % i)
        #elif i == 2:
        #    for j in range(2):
        #        x = slim.conv2d(x, c / 4, [1, 1],
        #                        scope='landmark_enc_pred_%d_blk_%d_conv1x1' % (i, j))
        #        x = slim.conv2d(x, c / 2, [3, 3],
        #                        scope='landmark_enc_pred_%d_blk_%d_conv3x3' % (i, j))
        #else:
        #    for j in range(2):
        #        x = slim.conv2d(x, c / 2, [1, 1],
        #                        scope='landmark_enc_pred_%d_blk_%d_conv1x1' % (i, j))
        #        x = slim.conv2d(x, c, [3, 3],
        #                        scope='landmark_enc_pred_%d_blk_%d_conv3x3' % (i, j))
        x = slim.conv2d(x, c, [3, 3], scope='landmark_enc_pred_%d_blk_0_conv3x3' % i)
    return x



