import tensorflow._api.v2.compat.v1 as tf
from config import params

import random
from src.constants import SHUFFLE_BUFFER_SIZE, NUM_THREADS, RESIZE_METHOD
from src.input_pipline.random_image_crop import random_image_crop
from src.input_pipline.other_augmentations import random_color_manipulations,\
    random_flip_left_right, random_pixel_value_scale, random_jitter_boxes, random_rotation_change, random_brightness


class Pipeline:
    """Input pipeline for training or evaluating object detectors."""

    def __init__(self, filenames, batch_size, image_size,
                 repeat=False, shuffle=False, augmentation=False):
        """
        Note: when evaluating set batch_size to 1.

        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            batch_size: an integer.
            image_size: a list with two integers [width, height] or None,
                images of this size will be in a batch.
                If value is None then images will not be resized.
                In this case batch size must be 1.
            repeat: a boolean, whether repeat indefinitely.
            shuffle: whether to shuffle the dataset.
            augmentation: whether to do data augmentation.
        """
        if image_size is not None:
            self.image_width, self.image_height = image_size
            self.resize = True
        else:
            assert batch_size == 1
            self.image_width, self.image_height = None, None
            self.resize = False

        self.augmentation = augmentation
        self.batch_size = batch_size

        def get_num_samples(filename):
            return sum(1 for _ in tf.python_io.tf_record_iterator(filename))

        num_examples = 0
        for filename in filenames:
            num_examples_in_file = get_num_samples(filename)
            assert num_examples_in_file > 0
            num_examples += num_examples_in_file
        self.num_examples = num_examples
        assert self.num_examples > 0

        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        num_shards = len(filenames)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=num_shards)

        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.prefetch(buffer_size=batch_size)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.repeat(None if repeat else 1)

        if params['input_pipeline_params']['use_bbox_only']:
            dataset = dataset.map(self._parse_and_preprocess_only_det, num_parallel_calls=NUM_THREADS)
        else:
            dataset = dataset.map(self._parse_and_preprocess, num_parallel_calls=NUM_THREADS)

        # we need batches of fixed size
        padded_shapes = (
            [],  # filename
            [self.image_height, self.image_width, 3],
            [None, 4],  # boxes
            [],  # num_boxes
            [None, 42],  # landmarks
            [None],  # labels
            []  # small_object
        )
        dataset = dataset.apply(
           tf.contrib.data.padded_batch_and_drop_remainder(batch_size, padded_shapes)
        )
        dataset = dataset.prefetch(buffer_size=1)

        self.iterator = dataset.make_one_shot_iterator()

    def get_batch(self):
        """
        Returns:
            features: a dict with the following keys
                'images': a float tensor with shape [batch_size, image_height, image_width, 3].
                'filenames': a string tensor with shape [batch_size].
            labels: a dict with the following keys
                'boxes': a float tensor with shape [batch_size, max_num_boxes, 4].
                'num_boxes': an int tensor with shape [batch_size].
            where max_num_boxes = max(num_boxes).
        """
        filenames, images, boxes, num_boxes, \
        landmarks, labels, \
        num_small_object = self.iterator.get_next()

        if self.augmentation and params["model_params"]["use_stitcher"]:
            small_object_ratio = self.calc_small_object_raio(num_boxes, num_small_object)
            ratio_thresh = 0.1
            need_to_stitch = tf.less(small_object_ratio, ratio_thresh)

            gt_info = (filenames, images, boxes, num_boxes, landmarks, labels)

            gt_info = \
                tf.cond(
                    need_to_stitch,
                    lambda: self.stitch(gt_info),
                    lambda: gt_info
                )

            filenames, images, boxes, num_boxes, landmarks, labels = gt_info

        features = {'images': images, 'filenames': filenames}
        labels = {
            'boxes': boxes,
            'num_boxes': num_boxes,
            'landmarks': landmarks,
            'gesture_labels': labels
        }
        return features, labels


    def calc_small_object_raio(self, num_boxes, num_small_object):
        return tf.reduce_sum(tf.to_float(num_small_object)) / tf.reduce_sum(tf.to_float(num_boxes))

    def stitch(self, gt_info):
        """
        stitch 4 square img to one big img then resize it to org size
        :param gt_info:
        :return:
        """
        filenames0, images0, boxes0, num_boxes0, landmarks0, labels0 = gt_info
        filenames1, images1, boxes1, num_boxes1, landmarks1, labels1, _ = self.iterator.get_next()
        filenames2, images2, boxes2, num_boxes2, landmarks2, labels2, _ = self.iterator.get_next()
        filenames3, images3, boxes3, num_boxes3, landmarks3, labels3, _ = self.iterator.get_next()

        boxes_list = [boxes0, boxes1, boxes2, boxes3]
        num_boxes_list = [num_boxes0, num_boxes1, num_boxes2, num_boxes3]
        landmarks_list = [landmarks0, landmarks1, landmarks2, landmarks3]

        def expand(x):
            return tf.expand_dims(x, axis=-1)

        # occlude_list = [occlude0, occlude1, occlude2, occlude3]
        # blur_list = [expand(blur0), expand(blur1), expand(blur2), expand(blur3)]
        # quality_list = [expand(quality0), expand(quality1), expand(quality2), expand(quality3)]
        labels_list = [expand(labels0), expand(labels1), expand(labels2), expand(labels3)]

        images = self.stitch_images(images0, images1, images2, images3)

        boxes, num_boxes = self.stitch_boxes(boxes_list, num_boxes_list)

        landmarks, _ = self.stitch_landmarks(landmarks_list, num_boxes_list)

        labels, _ = self.merge_for_stitch_int(labels_list, num_boxes_list)

        labels = tf.squeeze(labels, axis=-1)


        # occlude, _ = self.merge_for_stitch(occlude_list, num_boxes_list)

        # blur, _ = self.merge_for_stitch(blur_list, num_boxes_list)
        # blur = tf.squeeze(blur, axis=-1)

        # quality, _ = self.merge_for_stitch(quality_list, num_boxes_list)
        # quality = tf.squeeze(quality, axis=-1)

        return filenames0, images, boxes, num_boxes, landmarks, labels

    def stitch_images(self, images0, images1, images2, images3):
        img_h = tf.shape(images0)[1]
        img_w = tf.shape(images0)[2]
        images0 = tf.image.pad_to_bounding_box(images0, 0, 0, 2*img_h, 2*img_w)
        images1 = tf.image.pad_to_bounding_box(images1, 0, img_w, 2*img_h, 2*img_w)
        images2 = tf.image.pad_to_bounding_box(images2, img_h, 0, 2*img_h, 2*img_w)
        images3 = tf.image.pad_to_bounding_box(images3, img_h, img_w, 2*img_h, 2*img_w)
        images = images0 + images1 + images2 + images3
        images = tf.image.resize_images(
            images, [self.image_height, self.image_width],
            method=RESIZE_METHOD
        )
        return images

    def stitch_boxes(self, boxes_list, num_boxes_list):
        boxes0, boxes1, boxes2, boxes3 = boxes_list
        batch_size = boxes0.shape[0].value

        boxes0 = boxes0 / 2.0
        boxes3 = boxes3 / 2.0 + 0.5

        offset_boxes1 = [[[0, 0.5, 0, 0.5]]]  # ymin, xmin, ymax, xmax
        offset_boxes1 = tf.tile(offset_boxes1, [batch_size, tf.shape(boxes1)[1], 1])
        boxes1 = boxes1 / 2.0 + offset_boxes1

        offset_boxes2 = [[[0.5, 0, 0.5, 0]]]  # ymin, xmin, ymax, xmax
        offset_boxes2 = tf.tile(offset_boxes2, [batch_size, tf.shape(boxes2)[1], 1])
        boxes2 = boxes2 / 2.0 + offset_boxes2

        boxes, num_boxes = self.merge_for_stitch([boxes0, boxes1, boxes2, boxes3], num_boxes_list)

        return boxes, num_boxes

    def stitch_landmarks(self, landmarks_list, num_landmarks_list):
        landmarks0, landmarks1, landmarks2, landmarks3 = landmarks_list
        batch_size = landmarks0.shape[0].value

        landmarks0 = landmarks0 / 2.0
        landmarks3 = landmarks3 / 2.0 + 0.5

        offset_weight_1 = []
        counts = 0
        while counts <= 40:
            offset_weight_1.append(0.5)
            offset_weight_1.append(0)
            counts += 2
        offset_weight_2 = []
        counts = 0
        while counts <= 40:
            offset_weight_2.append(0)
            offset_weight_2.append(0.5)
            counts += 2
        # print(offset_weight_1)
        # print(offset_weight_2)
        # exit(0)

        offset_landmarks1 = [[offset_weight_1]]
        offset_landmarks1 = tf.tile(offset_landmarks1, [batch_size, tf.shape(landmarks1)[1], 1])
        landmarks1 = landmarks1 / 2.0 + offset_landmarks1

        offset_landmarks2 = [[offset_weight_2]]
        offset_landmarks2 = tf.tile(offset_landmarks2, [batch_size, tf.shape(landmarks2)[1], 1])
        landmarks2 = landmarks2 / 2.0 + offset_landmarks2

        landmarks, num_landmarks = self.merge_for_stitch([landmarks0, landmarks1, landmarks2, landmarks3],
                                                         num_landmarks_list)

        return landmarks, num_landmarks

    def merge_for_stitch(self, boxes_list, num_boxes_list):
        boxes0, boxes1, boxes2, boxes3 = boxes_list
        num_boxes0, num_boxes1, num_boxes2, num_boxes3 = num_boxes_list

        batch_size = boxes0.shape[0].value

        boxes0_list = tf.split(boxes0, num_or_size_splits=batch_size, axis=0)
        boxes1_list = tf.split(boxes1, num_or_size_splits=batch_size, axis=0)
        boxes2_list = tf.split(boxes2, num_or_size_splits=batch_size, axis=0)
        boxes3_list = tf.split(boxes3, num_or_size_splits=batch_size, axis=0)
        boxes_list = []
        num_boxes_list = []
        max_num_box = tf.shape(boxes0)[1] + tf.shape(boxes1)[1] + tf.shape(boxes2)[1] + tf.shape(boxes3)[1]
        for batch_idx in range(batch_size):
            boxes0_one_batch = boxes0_list[batch_idx][:, :num_boxes0[batch_idx], :]
            boxes1_one_batch = boxes1_list[batch_idx][:, :num_boxes1[batch_idx], :]
            boxes2_one_batch = boxes2_list[batch_idx][:, :num_boxes2[batch_idx], :]
            boxes3_one_batch = boxes3_list[batch_idx][:, :num_boxes3[batch_idx], :]
            num_boxes = num_boxes0[batch_idx] + num_boxes1[batch_idx] + num_boxes2[batch_idx] + num_boxes3[batch_idx]
            num_boxes_list.append(tf.expand_dims(num_boxes, axis=0))
            pad_size = max_num_box - num_boxes
            pad_tensor = tf.zeros([1, pad_size, tf.shape(boxes0)[2]])
            boxes_one_batch = tf.concat([
                boxes0_one_batch,
                boxes1_one_batch,
                boxes2_one_batch,
                boxes3_one_batch,
                pad_tensor
            ], axis=1)
            boxes_list.append(boxes_one_batch)
        boxes = tf.concat(boxes_list, axis=0)
        num_boxes = tf.concat(num_boxes_list, axis=0)
        return boxes, num_boxes


    def merge_for_stitch_int(self, boxes_list, num_boxes_list):
        boxes0, boxes1, boxes2, boxes3 = boxes_list
        num_boxes0, num_boxes1, num_boxes2, num_boxes3 = num_boxes_list

        batch_size = boxes0.shape[0].value

        boxes0_list = tf.split(boxes0, num_or_size_splits=batch_size, axis=0)
        boxes1_list = tf.split(boxes1, num_or_size_splits=batch_size, axis=0)
        boxes2_list = tf.split(boxes2, num_or_size_splits=batch_size, axis=0)
        boxes3_list = tf.split(boxes3, num_or_size_splits=batch_size, axis=0)
        boxes_list = []
        num_boxes_list = []
        max_num_box = tf.shape(boxes0)[1] + tf.shape(boxes1)[1] + tf.shape(boxes2)[1] + tf.shape(boxes3)[1]
        for batch_idx in range(batch_size):
            boxes0_one_batch = boxes0_list[batch_idx][:, :num_boxes0[batch_idx], :]
            boxes1_one_batch = boxes1_list[batch_idx][:, :num_boxes1[batch_idx], :]
            boxes2_one_batch = boxes2_list[batch_idx][:, :num_boxes2[batch_idx], :]
            boxes3_one_batch = boxes3_list[batch_idx][:, :num_boxes3[batch_idx], :]
            num_boxes = num_boxes0[batch_idx] + num_boxes1[batch_idx] + num_boxes2[batch_idx] + num_boxes3[batch_idx]
            num_boxes_list.append(tf.expand_dims(num_boxes, axis=0))
            pad_size = max_num_box - num_boxes
            pad_tensor = tf.zeros([1, pad_size, tf.shape(boxes0)[2]], dtype=tf.int32)
            boxes_one_batch = tf.concat([
                boxes0_one_batch,
                boxes1_one_batch,
                boxes2_one_batch,
                boxes3_one_batch,
                pad_tensor
            ], axis=1)
            boxes_list.append(boxes_one_batch)
        boxes = tf.concat(boxes_list, axis=0)
        num_boxes = tf.concat(num_boxes_list, axis=0)
        return boxes, num_boxes


    def _parse_and_preprocess(self, example_proto):
        """What this function does:
        1. Parses one record from a tfrecords file and decodes it.
        2. (optionally) Augments it.

        Returns:
            image: a float tensor with shape [image_height, image_width, 3],
                an RGB image with pixel values in the range [0, 1].
            boxes: a float tensor with shape [num_boxes, 4].
            num_boxes: an int tensor with shape [].
            filename: a string tensor with shape [].
        """
        def landmark_re(num):
            dict = {}
            for i in range(num):
                name = f'landmark_{i}_x'
                dict[name] = tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
                name = f'landmark_{i}_y'
                dict[name] = tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
            return dict

        features_1 = {
            'filename': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
            'ymin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'ymax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'gesture_labels': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),

        }
        features_2 = landmark_re(21)
        features = {**features_1, **features_2}
        # print(features)
        parsed_features = tf.parse_single_example(example_proto, features)

        # get image
        image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # now pixel values are scaled to [0, 1] range

        # boxes
        # get groundtruth boxes, they must be in from-zero-to-one format
        boxes = tf.stack([
            parsed_features['ymin'], parsed_features['xmin'],
            parsed_features['ymax'], parsed_features['xmax']
        ], axis=1)
        boxes = tf.to_float(boxes)
        # it is important to clip here!
        boxes = tf.clip_by_value(boxes, clip_value_min=0.0, clip_value_max=1.0)

        def parsed_features_landmark_name():
            list_name = []
            for i in range(21):
                name_1 = f'landmark_{i}_x'
                list_name.append(parsed_features[name_1])
                name_2 = f'landmark_{i}_y'
                list_name.append(parsed_features[name_2])
            # print(list_name)
            # exit(0)
            return list_name
        # landmarks
        landmarks = tf.stack(parsed_features_landmark_name(), axis=1)
        landmarks = tf.to_float(landmarks)
        #landmark_shape = landmarks.shape.as_list()
        #landmarks = tf.zeros(landmark_shape)
        # it is important to clip here!
        landmarks = tf.clip_by_value(landmarks, clip_value_min=0.0, clip_value_max=1.0)

        labels = tf.stack(parsed_features['gesture_labels'], axis=0)
        labels = tf.to_int32(labels)
        '''
        # occlude
        occlude = tf.stack([
            parsed_features['lefteye_occlude'],
            parsed_features['righteye_occlude'],
            parsed_features['nose_occlude'],
            parsed_features['leftmouth_occlude'],
            parsed_features['rightmouth_occlude']
        ], axis=1)
        occlude = tf.to_float(occlude)

        # blur
        blur = parsed_features['blur']
        blur = tf.to_float(blur)
        blur = tf.clip_by_value(blur, clip_value_min=0.0, clip_value_max=1.0)

        # quality
        quality = parsed_features['quality']
        quality = tf.to_float(quality)
        quality = tf.clip_by_value(quality, clip_value_min=0.0, clip_value_max=1.0)
        #quality = tf.expand_dims(quality, axis=-1)  # expand dim to align prediction

        '''
        if self.augmentation:
            image, boxes, landmarks, labels = \
                self._augmentation_fn(image, boxes, landmarks, labels)
        else:
            image = tf.image.resize_images(
                image, [self.image_height, self.image_width],
                method=RESIZE_METHOD
            ) if self.resize else image

        #boxes, landmarks, occlude, blur, quality = \
        #    self.delete_small_boxes(boxes, landmarks, occlude, blur, quality)

        num_boxes = tf.to_int32(tf.shape(boxes)[0])
        filename = parsed_features['filename']

        # small object
        small_object_size = 12 / 512.0
        ymin, xmin, ymax, xmax = tf.split(boxes, num_or_size_splits=4, axis=-1)
        w = xmax - xmin
        h = ymax - ymin
        num_small_object = tf.reduce_sum(tf.to_int32(tf.less(w * h, small_object_size**2)))

        return filename, image, boxes, num_boxes, landmarks, labels, num_small_object

    def _parse_and_preprocess_only_det(self, example_proto):
        features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
            'ymin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'ymax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'gesture_labels': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # get image
        image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # now pixel values are scaled to [0, 1] range

        # boxes
        # get groundtruth boxes, they must be in from-zero-to-one format
        boxes = tf.stack([
            parsed_features['ymin'], parsed_features['xmin'],
            parsed_features['ymax'], parsed_features['xmax']
        ], axis=1)
        boxes = tf.to_float(boxes)
        # it is important to clip here!
        boxes = tf.clip_by_value(boxes, clip_value_min=0.0, clip_value_max=1.0)

        # landmarks
        n_boxes = tf.shape(boxes)[0]
        landmarks = tf.zeros([n_boxes, 42])
        labels = tf.zeros([n_boxes])
        '''
        # occlude
        occlude = tf.zeros([n_boxes, 5])

        # blur
        blur = tf.zeros([n_boxes])

        # quality
        quality = tf.zeros([n_boxes])
        '''

        if self.augmentation:
            image, boxes, landmarks, labels = \
                self._augmentation_fn(image, boxes, landmarks, labels)
        else:
            image = tf.image.resize_images(
                image, [self.image_height, self.image_width],
                method=RESIZE_METHOD
            ) if self.resize else image

        # boxes, landmarks, occlude, blur, quality = \
        #    self.delete_small_boxes(boxes, landmarks, occlude, blur, quality)

        num_boxes = tf.to_int32(tf.shape(boxes)[0])
        filename = parsed_features['filename']

        # small object
        small_object_size = 12 / 512.0
        ymin, xmin, ymax, xmax = tf.split(boxes, num_or_size_splits=4, axis=-1)
        w = xmax - xmin
        h = ymax - ymin
        num_small_object = tf.reduce_sum(tf.to_int32(tf.less(w * h, small_object_size ** 2)))

        return image, boxes, num_boxes, filename, landmarks, labels, num_small_object


    def _augmentation_fn(self, image, boxes, landmarks, labels):
        # there are a lot of hyperparameters here,
        # you will need to tune them all, haha
        image, boxes, landmarks = random_rotation_change(image, boxes, landmarks, max_angle=15)
        # '''
        image, boxes, landmarks, labels = \
            random_image_crop(
                image, boxes, landmarks, labels,
                probability=0.9,
                min_object_covered=0.9,
                #min_object_covered=0.4,
                aspect_ratio_range=(0.93, 1.07),
                area_range=(0.8, 1),
                overlap_thresh=0.4
            )
        # '''


        #candidate_size_list = [[256, 256], [320, 320], [384, 384], [448, 448], [512, 512], [576, 576],
        #                      [640, 640], [704, 704]]
        #rand_idx = random.randint(0, len(candidate_size_list)-1)
        #image_size = candidate_size_list[rand_idx]
        #self.image_height, self.image_width = image_size
        # '''
        image = tf.image.resize_images(
            image, [self.image_height, self.image_width],
            method=RESIZE_METHOD
        ) if self.resize else image
        # '''
        # image, boxes, landmarks = random_rotation_change(image, boxes, landmarks, max_angle=30)

        # if you do color augmentations before resizing, it will be very slow!
        # image = random_color_manipulations(image, probability=0.45, grayscale_probability=0.05)
        # image = random_pixel_value_scale(image, minval=0.85, maxval=1.15, probability=0.2)
        # boxes = random_jitter_boxes(boxes, landmarks, ratio=0.01)
        image = random_brightness(image, random_fraction=0.25, max_coefficient=0.2)
        image, boxes, landmarks = random_flip_left_right(image, boxes, landmarks)

        return image, boxes, landmarks, labels

    def delete_small_boxes(self, boxes, landmarks, labels):
        # small boxes may decrease performance
        with tf.name_scope('delete_small_boxes'):
            min_size = 8.0
            y_min, x_min, y_max, x_max = \
                tf.split(boxes, num_or_size_splits=4, axis=1)
            w = x_max - x_min
            h = y_max - y_min
            #keep_bool_w = tf.less(w, tf.constant(min_size))
            #keep_bool_h = tf.less(h, tf.constant(min_size))
            keep_bool_w = tf.greater(w, tf.constant(min_size))
            keep_bool_h = tf.greater(h, tf.constant(min_size))
            keep_bool = tf.logical_and(keep_bool_w, keep_bool_h)
            keep_bool = tf.squeeze(keep_bool, axis=-1)
            keep_inds = tf.squeeze(tf.where(keep_bool), axis=1)
            boxes = tf.gather(boxes, keep_inds)
            landmarks = tf.gather(landmarks, keep_inds)
            labels = tf.gather(labels, keep_inds)
            return boxes, landmarks, labels

