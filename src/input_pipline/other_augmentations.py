import tensorflow._api.v2.compat.v1 as tf
import math
import random

"""
There are various data augmentations for training object detectors.

`image` is assumed to be a float tensor with shape [height, width, 3],
it is a RGB image with pixel values in range [0, 1].
"""


def random_color_manipulations(
        image, probability=0.5, grayscale_probability=0.1):

    def manipulate(image):
        # intensity and order of this operations are kinda random,
        # so you will need to tune this for you problem
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_hue(image, 0.1)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def to_grayscale(image):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
        return image

    with tf.name_scope('random_color_manipulations'):
        do_it = tf.less(tf.random_uniform([]), probability)
        image = tf.cond(do_it, lambda: manipulate(image), lambda: image)

    with tf.name_scope('to_grayscale'):
        make_gray = tf.less(tf.random_uniform([]), grayscale_probability)
        image = tf.cond(make_gray, lambda: to_grayscale(image), lambda: image)

    return image


def random_brightness(image, random_fraction=0.5, max_coefficient=0.2):
    def adjust_brightness(img):
        bright_coefficient = tf.random_uniform([], 0, max_coefficient)
        img = tf.image.adjust_brightness(img, bright_coefficient)
        return img

    with tf.name_scope('random_brightness'):
        is_adjust_okay = tf.less(tf.random_uniform([]), random_fraction)
        image = tf.cond(is_adjust_okay,
                        lambda: adjust_brightness(image),
                        lambda: image, )

    return image


def random_flip_left_right(image, boxes, landmarks):

    def flip(image, boxes, landmarks):
        flipped_image = tf.image.flip_left_right(image)

        # box
        ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
        flipped_xmin = tf.subtract(1.0, xmax)
        flipped_xmax = tf.subtract(1.0, xmin)
        flipped_boxes = tf.stack([ymin, flipped_xmin, ymax, flipped_xmax], 1)

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

        flipped_landmark_0_x = tf.subtract(1.0, landmarks_0_x)
        flipped_landmark_1_x = tf.subtract(1.0, landmarks_1_x)
        flipped_landmark_2_x = tf.subtract(1.0, landmarks_2_x)
        flipped_landmark_3_x = tf.subtract(1.0, landmarks_3_x)
        flipped_landmark_4_x = tf.subtract(1.0, landmarks_4_x)
        flipped_landmark_5_x = tf.subtract(1.0, landmarks_5_x)
        flipped_landmark_6_x = tf.subtract(1.0, landmarks_6_x)
        flipped_landmark_7_x = tf.subtract(1.0, landmarks_7_x)
        flipped_landmark_8_x = tf.subtract(1.0, landmarks_8_x)
        flipped_landmark_9_x = tf.subtract(1.0, landmarks_9_x)
        flipped_landmark_10_x = tf.subtract(1.0, landmarks_10_x)
        flipped_landmark_11_x = tf.subtract(1.0, landmarks_11_x)
        flipped_landmark_12_x = tf.subtract(1.0, landmarks_12_x)
        flipped_landmark_13_x = tf.subtract(1.0, landmarks_13_x)
        flipped_landmark_14_x = tf.subtract(1.0, landmarks_14_x)
        flipped_landmark_15_x = tf.subtract(1.0, landmarks_15_x)
        flipped_landmark_16_x = tf.subtract(1.0, landmarks_16_x)
        flipped_landmark_17_x = tf.subtract(1.0, landmarks_17_x)
        flipped_landmark_18_x = tf.subtract(1.0, landmarks_18_x)
        flipped_landmark_19_x = tf.subtract(1.0, landmarks_19_x)
        flipped_landmark_20_x = tf.subtract(1.0, landmarks_20_x)

        flipped_landmark_0_y = landmarks_0_y
        flipped_landmark_1_y = landmarks_1_y
        flipped_landmark_2_y = landmarks_2_y
        flipped_landmark_3_y = landmarks_3_y
        flipped_landmark_4_y = landmarks_4_y
        flipped_landmark_5_y = landmarks_5_y
        flipped_landmark_6_y = landmarks_6_y
        flipped_landmark_7_y = landmarks_7_y
        flipped_landmark_8_y = landmarks_8_y
        flipped_landmark_9_y = landmarks_9_y
        flipped_landmark_10_y = landmarks_10_y
        flipped_landmark_11_y = landmarks_11_y
        flipped_landmark_12_y = landmarks_12_y
        flipped_landmark_13_y = landmarks_13_y
        flipped_landmark_14_y = landmarks_14_y
        flipped_landmark_15_y = landmarks_15_y
        flipped_landmark_16_y = landmarks_16_y
        flipped_landmark_17_y = landmarks_17_y
        flipped_landmark_18_y = landmarks_18_y
        flipped_landmark_19_y = landmarks_19_y
        flipped_landmark_20_y = landmarks_20_y

        flipped_landmarks = tf.stack([
            flipped_landmark_0_x, flipped_landmark_0_y,
            flipped_landmark_1_x, flipped_landmark_1_y,
            flipped_landmark_2_x, flipped_landmark_2_y,
            flipped_landmark_3_x, flipped_landmark_3_y,
            flipped_landmark_4_x, flipped_landmark_4_y,
            flipped_landmark_5_x, flipped_landmark_5_y,
            flipped_landmark_6_x, flipped_landmark_6_y,
            flipped_landmark_7_x, flipped_landmark_7_y,
            flipped_landmark_8_x, flipped_landmark_8_y,
            flipped_landmark_9_x, flipped_landmark_9_y,
            flipped_landmark_10_x, flipped_landmark_10_y,
            flipped_landmark_11_x, flipped_landmark_11_y,
            flipped_landmark_12_x, flipped_landmark_12_y,
            flipped_landmark_13_x, flipped_landmark_13_y,
            flipped_landmark_14_x, flipped_landmark_14_y,
            flipped_landmark_15_x, flipped_landmark_15_y,
            flipped_landmark_16_x, flipped_landmark_16_y,
            flipped_landmark_17_x, flipped_landmark_17_y,
            flipped_landmark_18_x, flipped_landmark_18_y,
            flipped_landmark_19_x, flipped_landmark_19_y,
            flipped_landmark_20_x, flipped_landmark_20_y,
            ], axis=1)

        return flipped_image, flipped_boxes, flipped_landmarks

    with tf.name_scope('random_flip_left_right'):
        do_it = tf.less(tf.random_uniform([]), 0.5)
        image, boxes, landmarks = \
            tf.cond(do_it,
                    lambda: flip(image, boxes, landmarks),
                    lambda: (image, boxes, landmarks))
        return image, boxes, landmarks


def random_pixel_value_scale(image, minval=0.9, maxval=1.1, probability=0.5):
    """This function scales each pixel independently of the other ones.

    Arguments:
        image: a float tensor with shape [height, width, 3],
            an image with pixel values varying between [0, 1].
        minval: a float number, lower ratio of scaling pixel values.
        maxval: a float number, upper ratio of scaling pixel values.
        probability: a float number.
    Returns:
        a float tensor with shape [height, width, 3].
    """
    def random_value_scale(image):
        color_coefficient = tf.random_uniform(
            tf.shape(image), minval=minval,
            maxval=maxval, dtype=tf.float32
        )
        image = tf.multiply(image, color_coefficient)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    with tf.name_scope('random_pixel_value_scale'):
        do_it = tf.less(tf.random_uniform([]), probability)
        image = tf.cond(
            do_it,
            lambda: random_value_scale(image),
            lambda: image)
        return image


def random_jitter_boxes(boxes, landmarks, ratio=0.05):
    """Randomly jitter bounding boxes.

    Arguments:
        boxes: a float tensor with shape [N, 4].
        ratio: a float number.
            The ratio of the box width and height that the corners can jitter.
            For example if the width is 100 pixels and ratio is 0.05,
            the corners can jitter up to 5 pixels in the x direction.
    Returns:
        a float tensor with shape [N, 4].
    """
    def random_jitter_box(box_landmark, ratio):
        """Randomly jitter a box.
        Arguments:
            box: a float tensor with shape [4].
            ratio: a float number.
        Returns:
            a float tensor with shape [4].
        """
        def get_box(box_landmark):
            ymin, xmin, ymax, xmax = [box_landmark[i] for i in range(4)]
            box = [ymin, xmin, ymax, xmax]
            return box

        def get_box_with_landmark(box_landmark): # TODO: This may not be necessary
            ymin, xmin, ymax, xmax, \
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
            landmarks_20_x, landmarks_20_y = [box_landmark[i] for i in range(46)]





            # make sure boxes larger than landmarks

            landmark_xmin = tf.reduce_min(
                [landmarks_0_x, landmarks_1_x, landmarks_2_x, landmarks_3_x, landmarks_4_x, landmarks_5_x,
                 landmarks_6_x, landmarks_7_x, landmarks_8_x, landmarks_9_x, landmarks_10_x, landmarks_11_x,
                 landmarks_12_x, landmarks_13_x, landmarks_14_x, landmarks_15_x, landmarks_16_x, landmarks_17_x,
                 landmarks_18_x, landmarks_19_x, landmarks_20_x, ]
            )
            landmark_xmax = tf.reduce_max(
                [landmarks_0_x, landmarks_1_x, landmarks_2_x, landmarks_3_x, landmarks_4_x, landmarks_5_x,
                landmarks_6_x, landmarks_7_x, landmarks_8_x, landmarks_9_x, landmarks_10_x, landmarks_11_x,
                landmarks_12_x, landmarks_13_x, landmarks_14_x, landmarks_15_x, landmarks_16_x, landmarks_17_x,
                landmarks_18_x, landmarks_19_x, landmarks_20_x,]
            )
            landmark_ymin = tf.reduce_min(
                [landmarks_0_y, landmarks_1_y, landmarks_2_y, landmarks_3_y, landmarks_4_y, landmarks_5_y,
                 landmarks_6_y, landmarks_7_y, landmarks_8_y, landmarks_9_y, landmarks_10_y, landmarks_11_y,
                 landmarks_12_y, landmarks_13_y, landmarks_14_y, landmarks_15_y, landmarks_16_y, landmarks_17_y,
                 landmarks_18_y, landmarks_19_y, landmarks_20_y, ]
            )
            landmark_ymax = tf.reduce_max(
                [landmarks_0_y, landmarks_1_y, landmarks_2_y, landmarks_3_y, landmarks_4_y, landmarks_5_y,
                 landmarks_6_y, landmarks_7_y, landmarks_8_y, landmarks_9_y, landmarks_10_y, landmarks_11_y,
                 landmarks_12_y, landmarks_13_y, landmarks_14_y, landmarks_15_y, landmarks_16_y, landmarks_17_y,
                 landmarks_18_y, landmarks_19_y, landmarks_20_y, ]
            )



            ymin = tf.reduce_min([ymin, landmark_ymin])
            xmin = tf.reduce_min([xmin, landmark_xmin])
            ymax = tf.reduce_max([ymax, landmark_ymax])
            xmax = tf.reduce_max([xmax, landmark_xmax])

            box = [ymin, xmin, ymax, xmax]
            return box

        lefteye_x = box_landmark[4]
        box = tf.cond(lefteye_x > 0.0,
                      lambda: get_box_with_landmark(box_landmark),
                      lambda: get_box(box_landmark)
                      )

        ymin, xmin, ymax, xmax = box
        box_height, box_width = ymax - ymin, xmax - xmin
        hw_coefs = tf.stack([box_height, box_width, box_height, box_width])

        rand_numbers = tf.random_uniform(
            [4], minval=-ratio, maxval=ratio, dtype=tf.float32
        )
        hw_rand_coefs = tf.multiply(hw_coefs, rand_numbers)

        jittered_box = tf.add(box, hw_rand_coefs)
        return jittered_box

    with tf.name_scope('random_jitter_boxes'):
        boxes_landmarks = tf.concat([boxes, landmarks], axis=1)
        distorted_boxes = tf.map_fn(
            lambda x: random_jitter_box(x, ratio),
            boxes_landmarks, dtype=tf.float32, back_prop=False
        )
        distorted_boxes = tf.clip_by_value(distorted_boxes, 0.0, 1.0)

        return distorted_boxes



def random_rotation_change(image, boxes, landmarks, max_angle=15):
    def rotation_manipulation_box(box):
        box = tf.matmul((tf.reshape(box, [4, 2]) * scaler - center), rotation_matrix) + center
        y, x = tf.unstack(box / scaler, axis=1)
        ymin, ymax = tf.reduce_min(y), tf.reduce_max(y)
        xmin, xmax = tf.reduce_min(x), tf.reduce_max(x)
        ymin = tf.clip_by_value(ymin, 0.0, 1.0)
        xmin = tf.clip_by_value(xmin, 0.0, 1.0)
        ymax = tf.clip_by_value(ymax, 0.0, 1.0)
        xmax = tf.clip_by_value(xmax, 0.0, 1.0)
        box = tf.stack([ymin, xmin, ymax, xmax], axis=0)
        return box

    def rotation_manipulation_lmt(landmarks):
        landmarks = tf.reshape(landmarks, [21, 2])
        landmarks = tf.matmul(landmarks * scaler - center, rotation_matrix) + center
        landmarks = landmarks / scaler
        landmarks = tf.reshape(landmarks, [42])
        return landmarks


    with tf.name_scope('random_rotation'):
        # get a random angle
        max_angle_radians = max_angle * (math.pi / 180.0)
        theta = tf.random_uniform(
            [], minval=-max_angle_radians,
            maxval=max_angle_radians, dtype=tf.float32
        )
        # theta = max_angle_radians
        N_lmt = tf.shape(landmarks)[0]
        # N_lmt = landmarks.get_shape()[0]
        # print(N_lmt)
        N_box = tf.shape(boxes)[0]
        # N_box = boxes.get_shape()[0]
        # print(N_box)

        # find the center of the image
        image_height = tf.to_float(tf.shape(image)[0])
        image_width = tf.to_float(tf.shape(image)[1])
        scaler = tf.stack([image_height, image_width], axis=0)
        # print('scaler:', tf.shape(scaler))
        center = tf.reshape(0.5 * scaler, [1, 2])
        # print('center:', tf.shape(center))

        rotation = tf.stack([
            tf.cos(theta), tf.sin(theta),
            -tf.sin(theta), tf.cos(theta)
        ], axis=0)
        rotation_matrix = tf.reshape(rotation, [2, 2])

        inverse_rotation = tf.stack([
            tf.cos(theta), -tf.sin(theta),
            tf.sin(theta), tf.cos(theta)
        ], axis=0)
        inverse_rotation_matrix = tf.reshape(inverse_rotation, [2, 2])

        # now i want to rotate the image and annotations around the image center,
        # note: landmark and box coordinates are (y, x) not (x, y)
        # print(tf.to_int32(tf.shape(landmarks)[0]))

        # rotate box
        ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
        # ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=0)
        h, w = ymax - ymin, xmax - xmin
        boxes = tf.stack([
            ymin, xmin, ymin, xmax,
            ymax, xmax, ymax, xmin
        ], axis=1)  # four corners

        # boxes = tf.map_fn(
        #     lambda x: rotation_manipulation_box(boxes),
        #     boxes, dtype=tf.float32, back_prop=False
        # )
        #
        boxes = tf.map_fn(
            rotation_manipulation_box,
            boxes, dtype=tf.float32, back_prop=False
        )
        # boxes = tf.map_fn(
        #     lambda x: tf.matmul((tf.reshape(boxes, [4, 2]) * scaler - center), rotation_matrix) + center,
        #     boxes, dtype=tf.float32, back_prop=False
        # )
        # boxes = tf.map_fn(
        #     lambda x: boxes * scaler - center,
        #     boxes, dtype=tf.float32, back_prop=False
        # )
        # boxes = tf.map_fn(
        #     lambda x: tf.matmul(boxes, rotation_matrix) + center,
        #     boxes, dtype=tf.float32, back_prop=False
        # )

        # box = tf.matmul(boxes * scaler - center, rotation_matrix) + center
        # y, x = tf.map_fn(
        #     lambda x: tf.unstack(boxes / scaler, axis=1),
        #     boxes, dtype=tf.float32, back_prop=False
        # )
        # y, x = tf.unstack(boxes / scaler, axis=2)
        # ymin, ymax = tf.reduce_min(y), tf.reduce_max(y)
        # xmin, xmax = tf.reduce_min(x), tf.reduce_max(x)
        # box = tf.stack([ymin, xmin, ymax, xmax], axis=0)

        # ymin = tf.map_fn(
        #     lambda y: tf.reduce_min(y),
        #     y, dtype=tf.float32, back_prop=False
        # )
        # ymax = tf.map_fn(
        #     lambda y: tf.reduce_max(y),
        #     y, dtype=tf.float32, back_prop=False
        # )
        # xmin = tf.map_fn(
        #     lambda x: tf.reduce_min(x),
        #     x, dtype=tf.float32, back_prop=False
        # )
        # xmax = tf.map_fn(
        #     lambda x: tf.reduce_max(x),
        #     x, dtype=tf.float32, back_prop=False
        # )
        #
        # ymin = tf.clip_by_value(ymin, 0.0, 1.0)
        # xmin = tf.clip_by_value(xmin, 0.0, 1.0)
        # ymax = tf.clip_by_value(ymax, 0.0, 1.0)
        # xmax = tf.clip_by_value(xmax, 0.0, 1.0)
        #
        # boxes = tf.map_fn(
        #     lambda x: tf.stack([ymin, xmin, ymax, xmax], axis=0),
        #     [ymin, xmin, ymax, xmax], dtype=tf.float32, back_prop=False
        # )

        # box = tf.stack([ymin, xmin, ymax, xmax], axis=1)

        # rotate landmarks
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
        landmarks = tf.stack([landmarks_0_y, landmarks_0_x,
            landmarks_1_y, landmarks_1_x,
            landmarks_2_y, landmarks_2_x,
            landmarks_3_y, landmarks_3_x,
            landmarks_4_y, landmarks_4_x,
            landmarks_5_y, landmarks_5_x,
            landmarks_6_y, landmarks_6_x,
            landmarks_7_y, landmarks_7_x,
            landmarks_8_y, landmarks_8_x,
            landmarks_9_y, landmarks_9_x,
            landmarks_10_y, landmarks_10_x,
            landmarks_11_y, landmarks_11_x,
            landmarks_12_y, landmarks_12_x,
            landmarks_13_y, landmarks_13_x,
            landmarks_14_y, landmarks_14_x,
            landmarks_15_y, landmarks_15_x,
            landmarks_16_y, landmarks_16_x,
            landmarks_17_y, landmarks_17_x,
            landmarks_18_y, landmarks_18_x,
            landmarks_19_y, landmarks_19_x,
            landmarks_20_y, landmarks_20_x], axis=1)

        landmarks = tf.map_fn(
            rotation_manipulation_lmt,
            landmarks, dtype=tf.float32, back_prop=False
        )
        # landmarks = tf.map_fn(
        #     lambda x: tf.reshape(landmarks, [4, 2]),
        #     landmarks, dtype=tf.float32, back_prop=False
        # )
        # landmarks = tf.reshape(landmarks, [N_lmt, 4, 2])
        # landmarks = tf.map_fn(
        #     lambda x: tf.matmul(landmarks * scaler - center, rotation_matrix) + center,
        #     landmarks, dtype=tf.float32, back_prop=False
        # )
        # landmarks = tf.matmul(landmarks * scaler - center, rotation_matrix) + center
        # landmarks = tf.map_fn(
        #     lambda x: landmarks / scaler,
        #     landmarks, dtype=tf.float32, back_prop=False
        # )
        # landmarks = landmarks / scaler
        # landmarks = tf.map_fn(
        #     lambda x: tf.reshape(landmarks, [8]),
        #     landmarks, dtype=tf.float32, back_prop=False
        # )
        # landmarks = tf.reshape(landmarks, [21, 2])
        landmarks_0_y, landmarks_0_x,\
        landmarks_1_y, landmarks_1_x,\
        landmarks_2_y, landmarks_2_x,\
        landmarks_3_y, landmarks_3_x,\
        landmarks_4_y, landmarks_4_x,\
        landmarks_5_y, landmarks_5_x,\
        landmarks_6_y, landmarks_6_x,\
        landmarks_7_y, landmarks_7_x,\
        landmarks_8_y, landmarks_8_x,\
        landmarks_9_y, landmarks_9_x,\
        landmarks_10_y, landmarks_10_x,\
        landmarks_11_y, landmarks_11_x,\
        landmarks_12_y, landmarks_12_x,\
        landmarks_13_y, landmarks_13_x,\
        landmarks_14_y, landmarks_14_x,\
        landmarks_15_y, landmarks_15_x,\
        landmarks_16_y, landmarks_16_x,\
        landmarks_17_y, landmarks_17_x,\
        landmarks_18_y, landmarks_18_x,\
        landmarks_19_y, landmarks_19_x,\
        landmarks_20_y, landmarks_20_x = tf.unstack(landmarks, axis=1)

        landmarks_0_x = tf.clip_by_value(landmarks_0_x, 0.0, 1.0)
        landmarks_1_x = tf.clip_by_value(landmarks_1_x, 0.0, 1.0)
        landmarks_2_x = tf.clip_by_value(landmarks_2_x, 0.0, 1.0)
        landmarks_3_x = tf.clip_by_value(landmarks_3_x, 0.0, 1.0)
        landmarks_4_x = tf.clip_by_value(landmarks_4_x, 0.0, 1.0)
        landmarks_5_x = tf.clip_by_value(landmarks_5_x, 0.0, 1.0)
        landmarks_6_x = tf.clip_by_value(landmarks_6_x, 0.0, 1.0)
        landmarks_7_x = tf.clip_by_value(landmarks_7_x, 0.0, 1.0)
        landmarks_8_x = tf.clip_by_value(landmarks_8_x, 0.0, 1.0)
        landmarks_9_x = tf.clip_by_value(landmarks_9_x, 0.0, 1.0)
        landmarks_10_x = tf.clip_by_value(landmarks_10_x, 0.0, 1.0)
        landmarks_11_x = tf.clip_by_value(landmarks_11_x, 0.0, 1.0)
        landmarks_12_x = tf.clip_by_value(landmarks_12_x, 0.0, 1.0)
        landmarks_13_x = tf.clip_by_value(landmarks_13_x, 0.0, 1.0)
        landmarks_14_x = tf.clip_by_value(landmarks_14_x, 0.0, 1.0)
        landmarks_15_x = tf.clip_by_value(landmarks_15_x, 0.0, 1.0)
        landmarks_16_x = tf.clip_by_value(landmarks_16_x, 0.0, 1.0)
        landmarks_17_x = tf.clip_by_value(landmarks_17_x, 0.0, 1.0)
        landmarks_18_x = tf.clip_by_value(landmarks_18_x, 0.0, 1.0)
        landmarks_19_x = tf.clip_by_value(landmarks_19_x, 0.0, 1.0)
        landmarks_20_x = tf.clip_by_value(landmarks_20_x, 0.0, 1.0)

        landmarks_0_y = tf.clip_by_value(landmarks_0_y, 0.0, 1.0)
        landmarks_1_y = tf.clip_by_value(landmarks_1_y, 0.0, 1.0)
        landmarks_2_y = tf.clip_by_value(landmarks_2_y, 0.0, 1.0)
        landmarks_3_y = tf.clip_by_value(landmarks_3_y, 0.0, 1.0)
        landmarks_4_y = tf.clip_by_value(landmarks_4_y, 0.0, 1.0)
        landmarks_5_y = tf.clip_by_value(landmarks_5_y, 0.0, 1.0)
        landmarks_6_y = tf.clip_by_value(landmarks_6_y, 0.0, 1.0)
        landmarks_7_y = tf.clip_by_value(landmarks_7_y, 0.0, 1.0)
        landmarks_8_y = tf.clip_by_value(landmarks_8_y, 0.0, 1.0)
        landmarks_9_y = tf.clip_by_value(landmarks_9_y, 0.0, 1.0)
        landmarks_10_y = tf.clip_by_value(landmarks_10_y, 0.0, 1.0)
        landmarks_11_y = tf.clip_by_value(landmarks_11_y, 0.0, 1.0)
        landmarks_12_y = tf.clip_by_value(landmarks_12_y, 0.0, 1.0)
        landmarks_13_y = tf.clip_by_value(landmarks_13_y, 0.0, 1.0)
        landmarks_14_y = tf.clip_by_value(landmarks_14_y, 0.0, 1.0)
        landmarks_15_y = tf.clip_by_value(landmarks_15_y, 0.0, 1.0)
        landmarks_16_y = tf.clip_by_value(landmarks_16_y, 0.0, 1.0)
        landmarks_17_y = tf.clip_by_value(landmarks_17_y, 0.0, 1.0)
        landmarks_18_y = tf.clip_by_value(landmarks_18_y, 0.0, 1.0)
        landmarks_19_y = tf.clip_by_value(landmarks_19_y, 0.0, 1.0)
        landmarks_20_y = tf.clip_by_value(landmarks_20_y, 0.0, 1.0)

        landmarks = tf.stack([landmarks_0_x, landmarks_0_y,
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
        landmarks_20_x, landmarks_20_y], axis=1)

        # rotate image
        translate = center - tf.matmul(center, inverse_rotation_matrix)
        translate_y, translate_x = tf.unstack(tf.squeeze(translate, axis=0), axis=0)
        transform = tf.stack([
            tf.cos(theta), -tf.sin(theta), translate_x,
            tf.sin(theta), tf.cos(theta), translate_y,
            0.0, 0.0
        ])
        image = tf.contrib.image.transform(image, transform, interpolation='BILINEAR')

        return image, boxes, landmarks


