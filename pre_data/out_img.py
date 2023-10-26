import cv2
from tqdm import tqdm
import os
import random
import numpy as np

def draw_pic(image, landmarks_or):
    landmarks = np.reshape(np.array(landmarks_or), (-1, 2)).tolist()
    # print(len(landmarks))
    # print((landmarks[0], landmarks[1]), (landmarks[1], landmarks[2]))
    hand_connections = ([
        (landmarks[0], landmarks[1]),
        (landmarks[1], landmarks[2]),
        (landmarks[2], landmarks[3]),
        (landmarks[3], landmarks[4]),
        (landmarks[0], landmarks[5]),
        (landmarks[5], landmarks[6]),
        (landmarks[6], landmarks[7]),
        (landmarks[7], landmarks[8]),
        (landmarks[9], landmarks[10]),
        (landmarks[10], landmarks[11]),
        (landmarks[11], landmarks[12]),
        (landmarks[13], landmarks[14]),
        (landmarks[14], landmarks[15]),
        (landmarks[15], landmarks[16]),
        (landmarks[0], landmarks[17]),
        (landmarks[17], landmarks[18]),
        (landmarks[18], landmarks[19]),
        (landmarks[19], landmarks[20]),
        (landmarks[5], landmarks[9]),
        (landmarks[9], landmarks[13]),
        (landmarks[13], landmarks[17]),
    ])
    # print(hand_connections)

    for connection in hand_connections:
        cv2.line(image, (int(connection[0][0]), int(connection[0][1])),
                 (int(connection[1][0]), int(connection[1][1])), (0, 0, 255), 2)


def out_img(output_dir, input_dir):

    dirs_list = os.listdir(input_dir)
    for dir in tqdm(dirs_list):
        dir_path = os.path.join(input_dir, dir)
        if dir_path == '/Users/chenjiayi/Downloads/hand_dataset/hagrid/subsample/.DS_Store':
            continue
        # print(dir_path)
        filesname = [n for n in os.listdir(dir_path) if n.endswith('.jpg')]
        
        # continue
        for file in filesname:
            image_dir = os.path.join(dir_path, file)
            if os.path.exists(image_dir):
                image = cv2.imread(image_dir)

                if image is None:
                    continue
                else:
                    # image = cv2.resize(image, (288, 512))
                    cv2.imwrite(os.path.join(output_dir, file), image)
                # print(image_dir)
            else:
                print('not exist', image_dir)
                continue


def tran_img(input_dir, output_dir):
    img = os.listdir(input_dir)
    for i in img:
        pic_dir = os.path.join(input_dir, i)
        pic_dir = os.path.join(pic_dir, 'JPEGImages')
        image_list = os.listdir(pic_dir)
        for name in tqdm(image_list):
            img_dir = os.path.join(pic_dir, name)
            print(name)
            image = cv2.imread(img_dir)
            # cv2.imshow('', image)
            savename = os.path.join(output_dir, name)
            print(savename)
            cv2.imshow('', image)
            cv2.waitKey(0)
            # cv2.imwrite(savename, image)


def pick_img(input_dir, output_dir):
    img = os.listdir(input_dir)
    random_pick = random.sample(img, 256)
    for name in random_pick:
        pic_dir = os.path.join(input_dir, name)
        image = cv2.imread(pic_dir)
        cv2.imwrite(os.path.join(output_dir, name), image)

if __name__ == '__main__':
    # pick_img('/home/chenjy531/Desktop/data/trans/Final_hagrid', '/home/chenjy531/Desktop/data/trans/trans_hagrid')
    out_img('/Users/chenjiayi/Downloads/hand_dataset/hagrid/image', '/Users/chenjiayi/Downloads/hand_dataset/hagrid/subsample')
    # tran_img('/home/chenjy531/Desktop/data/chenjy/FINAL-HAGRID', '/home/chenjy531/Desktop/data/trans/Final_hagrid')