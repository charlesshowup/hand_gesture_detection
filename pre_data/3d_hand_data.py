import os
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np

def trans_data(frame_dir):
    list_dirs = [i for i in os.listdir(frame_dir) if i.endswith('txt')]
    # print(list_dirs)
    jpg_dirs = [os.path.join(frame_dir, i) for i in list_dirs if 'webcam' in i]
    bbox_list = [os.path.join(frame_dir, bbox) for bbox in list_dirs if 'bbox' in bbox]
    landmarks_list = [os.path.join(frame_dir, keypoints) for keypoints in list_dirs if 'jointsCam' in keypoints]
    data = pd.DataFrame()
    for i in jpg_dirs:
        print(i.split('/')[-1])
        with open(i, 'r') as f:
            content = f.readlines()
        for hang in tqdm(content):
            name_1, name_2 = hang.split('/')[-2:]
            name_2 = name_2.rstrip('\n')
            name = name_1 + '/' + name_2
            new_frame = []
            # sigle_data = pd.concat([sigle_data, pd.DataFrame(name)], axis=1)
            jpg_path = os.path.join(frame_dir, name)
            if cv2.imread(jpg_path) is not None:
                h, w, c = cv2.imread(jpg_path).shape
            else:
                print('the jpg is not exist')
                continue
            idx = name_1 + '/' + name_2.split('_')[0]
            idx = idx + '_'
            idx_last = name_2.split('_')[-1]
            idx_last = ('_' + idx_last).rstrip('.jpg')
            # print(sigle_data)
            # exit(0)
            for bbox in bbox_list: # open bbox frame // bbox: truth_path
                # print(bbox)
                with open(bbox, 'r') as bb:
                    bbox_content = bb.readlines()
                    for bounding_box in bbox_content: # bounding_box:fake path
                        data_name, frame_name_all = bounding_box.split('/')[-2:]
                        frame_name = data_name + '/' + frame_name_all.split('_')[0]
                        frame_name = frame_name + '_'
                        # print(frame_name)
                        # '''
                        frame_name_last = frame_name_all.split('_')[-1]

                        frame_name_last = '_' + frame_name_last.split('.')[0]
                        if (idx == frame_name) & (idx_last == frame_name_last): #match bbox_idx
                            # print(f'{idx}:::{frame_name}')

                            dir_path = os.path.join(frame_dir, os.path.join(bounding_box.split('/')[-2],
                                                                            bounding_box.split('/')[-1]))
                            # print(dir_path)
                            # print(name)
                            # print('***' * 8)
                            bbox_data = deal_bbox(dir_path.rstrip('\n'), h, w)
                            new_frame.append(name)
                            for i in bbox_data:
                                new_frame.append(i)
                            # pd_frame = pd.DataFrame(np.reshape(np.array(new_frame), (1, -1)))
                            # data = pd.concat([data, pd_frame], axis=0)

                        else:
                            continue
            for landmarks in landmarks_list:
                with open(landmarks, 'r') as ld:
                    landmarks_content = ld.readlines()

                    for landmark in landmarks_content:
                        data_name, frame_name_all = landmark.split('/')[-2:]
                        frame_name = data_name + '/' + frame_name_all.split('_')[0]
                        frame_name = frame_name + '_'
                        # new_frame = []
                        frame_name_last = frame_name_all.split('_')[-1]

                        frame_name_last = '_' + frame_name_last.split('.')[0]
                        if (idx == frame_name) & (idx_last == frame_name_last): #match bbox_idx
                            # print(f'{idx}:::{landmark}')
                            dir_path = os.path.join(frame_dir, os.path.join(landmark.split('/')[-2],
                                                                            landmark.split('/')[-1]))

                            landmark_data = deal_landmarks(dir_path.rstrip('\n'), h, w).tolist()
                            # print(landmark_data)
                            for i in landmark_data:
                                # print(i)
                                new_frame.append(i)

                            # print(new_frame)
                            # exit(0)
                            pd_frame = pd.DataFrame(np.reshape(np.array(new_frame), (1, -1)))
                            data = pd.concat([data, pd_frame], axis=0)
                        else:
                            continue



    data.to_csv('/home/chenjy531/Desktop/data/trans/hagrid/3d_multi_hand/annoation.txt',
                    sep=' ', header=None, index=None)
    print(data[0])
                        # '''



def deal_bbox(path, jpg_height, jpg_width,):
    # print(path)
    df = pd.read_csv(path, header=None, sep=' ')
    # df = df.rename(columns={'idx', 'location'})
    # print(df)
    landmarks = df[1].tolist()
    # landmarks = [int(i) for i in landmarks]
    # print(landmarks)
    x, y, width, height = landmarks[0], landmarks[1], landmarks[2], landmarks[3]
    width = float(width) - float(x)
    height = float(height) - float(y)
    return [x/jpg_width, y/jpg_height, width/jpg_width, height/jpg_height]


def deal_landmarks(path, jpg_height, jpg_width):
    df = pd.read_csv(path, header=None, sep=' ')
    keypoints_x = np.array(df[1].tolist())
    keypoints_y = np.array(df[2].tolist())
    norm_x = keypoints_x / jpg_width
    norm_y = keypoints_y / jpg_height
    keypoints = pd.concat([pd.DataFrame(np.reshape(norm_x, (-1, 1))),
                           pd.DataFrame(np.reshape(norm_y, (-1, 1)))], axis=1)

    keypoints = np.reshape(np.array(keypoints), (1, -1))
    # print(np.squeeze(keypoints).shape)
    return np.squeeze(keypoints)


def resize_jpg(input_dir, output_dst):
    input_name = os.listdir(input_dir)
    for name in tqdm(input_name):
        img_path = os.path.join(input_dir, name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (520, 520))
        cv2.imwrite(os.path.join(output_dst, name), img)
if __name__ == '__main__':
    # trans_data('/home/chenjy531/Desktop/data/chenjy/hand_gesture_datasets/multiview_hand_pose_dataset_release')
    input_dir = r'/home/chenjy531/Desktop/data/chenjy/hand_gesture_datasets/hands/RHD_v1/RHD_v1-1/RHD_published_v2/training/color'
    output_dst = r'/home/chenjy531/Desktop/data/trans/RHD_v1'
    resize_jpg(input_dir, output_dst)
