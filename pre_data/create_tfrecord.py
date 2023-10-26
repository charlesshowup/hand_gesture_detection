import tensorflow._api.v2.compat.v1.compat.v1 as tf
tf.disable_eager_execution()

import cv2
import numpy as np
import json
import shutil
import os
import math
from tqdm import tqdm
import random


def main():
    img_dir = r'/Users/chenjiayi/Downloads/hand_dataset/hagrid/image'
    # img_dir = r'/home/chenjy531/Desktop/data/chenjy/CelebA_official/Img/img_align_celeba_png.7z/img_align_celeba_png'
    ann_dir = r'/Users/chenjiayi/Downloads/hand_dataset/hagrid/ann_subsample'
    output_dir = '/Users/chenjiayi/Downloads/hand_gesture/dataset/train'
    num_shards = 1
    is_shuffle = True

    shutil.rmtree(output_dir, ignore_errors=True)
    os.mkdir(output_dir)

    annotation_list = getAnnoList(ann_dir, img_dir)
    # annotation_list = getNegList(img_dir)
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



def getNegList(img_dir):
    img_name_list = os.listdir(img_dir)
    annotation_list = []

    for name in tqdm(img_name_list[:3000]):
        img_name = os.path.join(img_dir, name)
        info_dict = {}
        info_dict['filename'] = img_name
        info_dict['num_boxes'] = 0
        info_dict['xmin_list'] = []
        info_dict['ymin_list'] = []
        info_dict['xmax_list'] = []
        info_dict['ymax_list'] = []
        for i in range(21):
            # print(i)
            info_dict[f'landmark_{i}_x'] = []
            info_dict[f'landmark_{i}_y'] = []
        info_dict['gesture_labels'] = []
        annotation_list.append(info_dict)

    print(annotation_list)
    return annotation_list

def getAnnoList(ann_dir, img_dir):
    json_list = [each for each in os.listdir(ann_dir) if each.endswith('.json')]
    json_list.sort()
    annotation_list = []
    for json_name in json_list:
        json_path = os.path.join(ann_dir, json_name)
        # print(json_name)
        with open(json_path, 'r') as fp:
            content = json.load(fp)
            # print(content)

            for i, k in tqdm(content.items()):
            # for i, k in content.items():
                # print(i)
                # print(k)
                # exit(0)
                info_dict = {}
                img_name = i + '.jpg'
                info_dict['filename'] = img_name
                # print(img_name)
                img_path = os.path.join(img_dir, img_name)
                img = cv2.imread(img_path)
                # cv2.imshow('',img)
                # cv2.waitKey(0)
                # h, w, _ = img.shape


                if not os.path.exists(img_path):
                    # print(img_path, 'not exist!')
                    continue
                bboxes_list = k['bboxes']
                num_box = len(bboxes_list)
                # print(np.array(bboxes_list).shape)
                # print(bboxes_list)
                info_dict['num_boxes'] = int(num_box)
                # info_dict['num_boxes'] = []
                info_dict['xmin_list'] = []
                info_dict['ymin_list'] = []
                info_dict['xmax_list'] = []
                info_dict['ymax_list'] = []
                for i in range(num_box):
                    line_list = bboxes_list[i]
                    x1 = float(line_list[0])
                    y1 = float(line_list[1])
                    x2 = float(line_list[2]) + x1
                    y2 = float(line_list[3]) + y1
                    info_dict['xmin_list'].append(x1)
                    info_dict['ymin_list'].append(y1)
                    info_dict['xmax_list'].append(x2)
                    info_dict['ymax_list'].append(y2)
                landmark_list = k['landmarks']
                num_landmarks = len(landmark_list)

                for i in range(21):
                    # print(i)
                    info_dict[f'landmark_{i}_x'] = []
                    info_dict[f'landmark_{i}_y'] = []
                # print(f'{num_landmarks}::::{num_box}')
                for i in range(num_landmarks):
                    landmark = np.squeeze(landmark_list[i])
                    if landmark.shape == (0, ):
                        m = 0
                        counts = 0
                        while m <= 40:
                            info_dict[f'landmark_{counts}_x'].append(-1.0)
                            info_dict[f'landmark_{counts}_y'].append(-1.0)
                            counts += 1
                            m += 2
                    else:
                        # print(landmark.shape)

                        landmark = np.reshape(landmark, (42, ))
                        num_landmarks_single = len(landmark)
                        m = 0
                        counts = 0
                        while m <= num_landmarks_single - 2:
                            info_dict[f'landmark_{counts}_x'].append(landmark[m])
                            info_dict[f'landmark_{counts}_y'].append(landmark[m + 1])
                            counts += 1
                            m += 2

                label_list = k['labels']
                info_dict['gesture_labels'] = []
                num_label = len(label_list)
                # print(num_label)
                # info_dict['num_labels'] = int(num_label)
                for i in range(num_label):
                    label = label_list[i]
                    hand_gesture = transfer_str_to_num(label)
                    info_dict['gesture_labels'].append(int(hand_gesture))

                annotation_list.append(info_dict)




    # exit(0)
    return annotation_list


def transfer_str_to_num(string):
    if string == 'call':
        return int(0)
    elif string == 'dislike':
        return int(1)
    elif string == 'fist':
        return int(2)
    elif string == 'four':
        return int(3)
    elif string == 'like':
        return int(4)
    elif string == 'mute':
        return int(5)
    elif string == 'ok':
        return int(6)
    elif string == 'one':
        return int(7)
    elif string == 'palm':
        return int(8)
    elif string == 'peace':
        return int(9)
    elif string == 'peace_inverted':
        return int(10)
    elif string == 'rock':
        return int(11)
    elif string == 'stop':
        return int(12)
    elif string == 'stop_inverted':
        return int(13)
    elif string == 'three':
        return int(14)
    elif string == 'three2':
        return int(15)
    elif string == 'two_up':
        return int(16)
    elif string == 'two_up_inverted':
        return int(17)
    elif string == 'no_gesture':
        return int(18)


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
