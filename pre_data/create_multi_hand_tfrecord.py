import tensorflow._api.v2.compat.v1 as tf
import pickle
import cv2
import numpy as np
import json
import shutil
import os
import math
from tqdm import tqdm
import random


def main():
    img_dir = r'/home/chenjy531/Desktop/data/trans/RHD_v1'
    # img_dir = r'/home/chenjy531/Desktop/data/chenjy/CelebA_official/Img/img_align_celeba_png.7z/img_align_celeba_png'
    ann_dir = r'/home/chenjy531/Desktop/data/chenjy/hand_gesture_datasets/hands/RHD_v1/RHD_v1-1/RHD_published_v2'
    output_dir = '/home/chenjy531/Desktop/data/trans/HAGRID_tfrecord/3d_train'
    num_shards = 1
    is_shuffle = True

    shutil.rmtree(output_dir, ignore_errors=True)
    os.mkdir(output_dir)

    annotation_list = getAnnoList(ann_dir)
    if is_shuffle:
        random.shuffle(annotation_list)

    num_examples = len(annotation_list)
    print('Number of images:', num_examples)

    shard_size = math.ceil(num_examples / num_shards)
    print('Number of images per shard:', shard_size)

    shard_id = 0
    num_examples_written = 0
    for annotation in tqdm(annotation_list):
        # print(annotation)
        if num_examples_written == 0:
            shard_path = os.path.join(
                output_dir,
                'shard-%04d.tfrecords' %
                shard_id)
            writer = tf.python_io.TFRecordWriter(shard_path)

        tf_example = dict_to_tf_example(annotation, img_dir)
        writer.write(tf_example.SerializeToString())
        num_examples_written += 1

        if num_examples_written == shard_size:
            shard_id += 1
            num_examples_written = 0
            writer.close()

    if num_examples_written != shard_size and num_examples % num_shards != 0:
        writer.close()

    print('Result is here:', output_dir)


def getAnnoList(ann_dir):
    # json_list = [each for each in os.listdir(ann_dir) if each.endswith('.json')]
    # json_list.sort()
    img_dir = r'/home/chenjy531/Desktop/data/chenjy/hand_gesture_datasets/hands/RHD_v1/RHD_v1-1/RHD_published_v2/training/color'
    with open(os.path.join(ann_dir, 'training/anno_training.pickle'), 'rb') as fi:
        anno_all = pickle.load(fi)
    annotation_list = []
    for sample_id, anno in tqdm(anno_all.items()):
        # print(single_data)
        break_out_flag = False
        info_dict = {}
        img_name = '%05d.png' % sample_id
        info_dict['filename'] = img_name
        img = cv2.imread(os.path.join(img_dir, img_name))
        h, w, _ = img.shape
        if not os.path.exists(os.path.join(img_dir, img_name)):
            print(os.path.join(img_dir, img_name), 'not exist!')
            continue
        landmarks = anno['uv_vis'][:, :2]

        kp_visible = anno['uv_vis'][:, 2] #check landmarks weather available
        kp_visible = np.reshape(np.array(kp_visible), (-1, 1))
        # print(landmarks.shape)
        # print(kp_visible.shape)
        landmarks = np.concatenate((landmarks, kp_visible), axis=1)
        keypoints_idx = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17]
        info_dict['xmin_list'] = []
        info_dict['ymin_list'] = []
        info_dict['xmax_list'] = []
        info_dict['ymax_list'] = []
        info_dict['gesture_labels'] = []
        for i in range(21):
            # print(i)
            info_dict[f'landmark_{i}_x'] = []
            info_dict[f'landmark_{i}_y'] = []
        num_hand = 0
        for i in np.split(landmarks, 2):
            keypoints = i[:, :2]
            # print(keypoints.shape)
            visible = i[:, -1]
            # print(visible)
            # print('**' * 4)
            # keypoints = i
            # print(i.shape)
            # print(np.count_nonzero(i))

            if np.count_nonzero(visible) < 19 and np.count_nonzero(visible) != 0:
                # print('**'*4)
                # print(np.count_nonzero(keypoints))
                break_out_flag = True
                break
            if np.count_nonzero(visible) == 0:
                continue

            else:
                bboxes_list = calc_bbox(keypoints, 20, h, w)
                num_hand += 1
                x1 = float(bboxes_list[0])
                y1 = float(bboxes_list[1])
                x2 = float(bboxes_list[2]) + x1
                y2 = float(bboxes_list[3]) + y1
                info_dict['xmin_list'].append(x1)
                info_dict['ymin_list'].append(y1)
                info_dict['xmax_list'].append(x2)
                info_dict['ymax_list'].append(y2)

                for n in range(len(keypoints)):
                    # print(keypoints_idx[n])
                    # print(keypoints[keypoints_idx[n]][0])
                    # exit(0)
                    # print(f'{keypoints[n][0]}::{keypoints[n][1]}')
                    landmark = keypoints[keypoints_idx[n]]
                    info_dict[f'landmark_{n}_x'].append(float(landmark[0] / w))
                    info_dict[f'landmark_{n}_y'].append(float(landmark[1] / h))

                    # cv2.circle(image, (int(keypoints[n][0]), int(keypoints[n][1])), radius=2,
                    #            color=(0, 255, 255), thickness=-1)
        info_dict['num_boxes'] = num_hand
        info_dict['gesture_labels'] = []
        # print(num_label)
        # info_dict['num_labels'] = int(num_label)
        for i in range(num_hand):
            hand_gesture = 18
            info_dict['gesture_labels'].append(18)
        # bboxes_list = calc_bbox(landmarks)
            # cv2.rectangle(image, (x_min, y_min), (x_min + width, y_min + height), (255, 255, 0))
        if break_out_flag:
            continue

        annotation_list.append(info_dict)
    # print(annotation_list)
    # exit(0)
    return annotation_list


def calc_bbox(keypoints, scale, h, w):
    keypoints = np.reshape(np.array(keypoints), (-1, 2))
    x_land, y_land = np.hsplit(keypoints, 2)
    x_min, x_max = np.min(x_land), np.max(x_land)
    y_min, y_max = np.min(y_land), np.max(y_land)

    x_min = int(x_min - scale)
    x_max = int(x_max + scale)
    y_min = int(y_min - scale)
    y_max = int(y_max + scale)

    width = x_max - x_min
    height = y_max - y_min
    return [x_min/w, y_min/h, width/w, height/h]


def dict_to_tf_example(annotation, image_dir):
    """Convert dict to tf.Example proto.

    Notice that this function normalizes the bounding
    box coordinates provided by the raw data.

    Arguments:
        data: a dict.
        image_dir: a string, path to the image directory.
    Returns:
        an instance of tf.Example.
    """
    image_name = annotation['filename']
    # print(image_name)
    assert image_name.endswith('.jpg') or image_name.endswith('.jpeg') or image_name.endswith('.png')

    image_path = os.path.join(image_dir, image_name)
    # print(image_path)
    with tf.gfile.GFile(image_path, 'rb') as f:
        encoded_jpg = f.read()

    # check image format
    # encoded_jpg_io = io.BytesIO(encoded_jpg)
    # image = PIL.Image.open(encoded_jpg_io)
    # if image.format != 'JPEG':
    #    raise ValueError('Image format not JPEG!')

    # img_width = int(annotation['size']['width'])
    # img_height = int(annotation['size']['height'])
    # assert img_width > 0 and img_height > 0
    # assert image.size[0] == img_width and image.size[1] == img_height

    if len(annotation['xmin_list']) == 0:
        print(image_name, 'is without any keypoints!')
    feature = {
        'filename': _bytes_feature(image_name.encode()),
        'image': _bytes_feature(encoded_jpg),
        'xmin': _float_list_feature(annotation['xmin_list']),
        'xmax': _float_list_feature(annotation['xmax_list']),
        'ymin': _float_list_feature(annotation['ymin_list']),
        'ymax': _float_list_feature(annotation['ymax_list']),
        'num_boxes': _int_list_feature_num(annotation['num_boxes']),

    }
    feature_2 = landmark_def(21, annotation)
    dict_1 = {'gesture_labels': _int_list_feature(annotation['gesture_labels'])}
    feature = {**feature, **feature_2}
    feature = {**feature, **dict_1}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # print(example)
    return example


def landmark_def(num, list_name):
    dict = {}
    # print(list_name)
    for i in range(num):
        for i in range(num):
            name = f'landmark_{i}_x'
            dict[name] = _float_list_feature(list_name[f'{name}'])
            name = f'landmark_{i}_y'
            dict[name] = _float_list_feature(list_name[f'{name}'])
        return dict


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _float_list_feature_key(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _int_list_feature_num(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == '__main__':
    main()
