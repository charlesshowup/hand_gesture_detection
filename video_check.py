import cv2
import os
import numpy as np
from gesture_landmark_labels_detector import GestureLandmarkDetector
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# MODEL_PATH = 'model_test_20221213_1.pb'
# MODEL_PATH = 'model_test_20230130.pb'
# MODEL_PATH = 'model_test_20230204.pb'
MODEL_PATH = 'model_test_20230227.pb'


def getProperSize(picWidth, picHeight):
    """
    keep ratio, long side
    :param picWidth:
    :param picHeight:
    :return:
    """
    NET_W, NET_H = 512, 288
    if picWidth >= picHeight:
        netInputW = NET_W
        ratio = netInputW/picWidth
        netInputH = int(ratio * picHeight)
    else:
        netInputH = NET_H
        ratio = netInputH/picHeight
        netInputW = int(ratio * picWidth)

    return netInputW, netInputH


def getvideo(filename):
    gesture_landmark_detector = GestureLandmarkDetector(
        MODEL_PATH,
        gpu_memory_fraction=0.85)
    videoCapture = cv2.VideoCapture(0)

    # get size and fps
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('/Users/chenjiayi/Downloads/fist.mp4', cv2.VideoWriter_fourcc(*'XVID'),
                                  int(fps), size)
    imgArr = []
    # READ FPS
    pause = False
    while True:
        success, frame = videoCapture.read()
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        if success is False:
            break
        # img = img.copy()
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        picHeight, picWidth, _ = imgRGB.shape
        netInputW, netInputH = getProperSize(picWidth, picHeight)
        imgRGB = cv2.resize(imgRGB, (netInputW, netInputH), interpolation=cv2.INTER_LINEAR)

        boxes, scores, landmarks = gesture_landmark_detector(imgRGB, score_threshold=0.85)
        boxes, landmarks = _normalized_to_image(frame, boxes, landmarks, netInputW, netInputH)
        draw_landmarks(frame, landmarks)
        frame_new = drawBoxes(frame, boxes, scores, landmarks, testing=True)

        # boxes, landmarks = _normalized_to_image(frame, boxes, landmarks, netInputW, netInputH)



        draw_landmarks(frame_new, landmarks)
        videoWriter.write(frame_new)
        imgArr.append(frame_new)
        # '''
        cv2.namedWindow('windows', 0)
        # cv2.resizeWindow('windows', 800, 800)
        cv2.imshow('windows', frame_new)

        # cv2.waitKey(int(3000 / int(fps)))
        key = cv2.waitKey(0 if pause else 1000 // int(fps))
        if key == ord('P') or key == ord('p'):
            pause = ~pause
        if key == ord('Q') or key == ord('q'):
            break

    for i in range(len(imgArr)):
        videoWriter.write(imgArr[i])

    videoCapture.release()


def _normalized_to_image(image, boxes, landmarks, width, height):
    h, w, c = image.shape
    keypoints = []
    bboxes = []
    # print(len(landmarks))
    if len(landmarks) != 0:
        for landmark in landmarks:
            for i in range(len(landmark)):
                if i // 2 == 0:
                    # print(landmarks[i])
                    keypoints.append((landmark[i]) / width * w)
                else:
                    keypoints.append((landmark[i]) / height * h)

        # print(keypoints)
        # exit(0)
        for box in boxes:
            for i in range(len(box)):
                if i // 2 == 0:
                    # print(landmarks[i])
                    bboxes.append((box[i]) / width * w)
                else:
                    bboxes.append((box[i]) / height * h)

        landmarks = np.reshape(np.array(keypoints), (-1, 42))
        boxes = np.reshape(np.array(bboxes), (-1, 4))
    else:
        landmarks = landmarks
        boxes = boxes

    return boxes, landmarks


def draw_landmarks(image, landmarks_or):
    if len(landmarks_or) != 0:
        for landmarks in landmarks_or:
            landmarks = np.reshape(np.array(landmarks), (-1, 2)).tolist()
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
            #     # Draws landmark points after finishing the connection lines, which is
            #     # aesthetically better.
            #     for landmark_px in idx_to_coordinates.values():
            #         cv2.circle(image, landmark_px, 2,
            #                    (0, 0, 255), -1)


def drawBoxes(img, boxes, scores, landmarks, testing):
    y1_array = boxes[:, 0]
    x1_array = boxes[:, 1]
    y2_array = boxes[:, 2]
    x2_array = boxes[:, 3]
    img_copy = img.copy()

    drawColor = (239, 209, 141)
    landmarks_list = []
    for i in range(x1_array.shape[0]):
        cv2.rectangle(
            img, (int(x1_array[i]), int(y1_array[i])), (int(x2_array[i]), int(y2_array[i])), drawColor, 3)
        pos = (int(x1_array[i] - 20), int(y1_array[i] - 20))
        score_string = '%.2f' % scores[i]
        if testing is True:
            # gesture_string = '%.2f' % np.argmax(labels[i], axis=-1)
            img = cv2.putText(img, 'labels:{}, pro:{}'.format('stop', score_string), pos, cv2.FONT_HERSHEY_PLAIN,
                              3, drawColor, 1)
            for j in range(21):
                x = landmarks[i][2*j]
                y = landmarks[i][2*j+1]
                landmarks_list.append((x, y))
                cv2.circle(img, center=(int(x), int(y)), radius=5, color=(0, 255, 255), thickness=-1)
        elif testing is False:
            gesture_string = 'others'
            if gesture_string == 12:
                gesture_string = 'stop'
            else:
                gesture_string = 'others'
            img = cv2.putText(img, 'labels:{}, pro:{}'.format(gesture_string, score_string), pos,
                              cv2.FONT_HERSHEY_COMPLEX, 1, drawColor, 1)
            # cv2.line(img, start_point, end_point, (0, 255, 255), 2)


    return img


def getProperSize(picWidth, picHeight):
    """
    keep ratio, long side
    :param picWidth:
    :param picHeight:
    :return:
    """
    NET_W, NET_H = 512, 288
    if picWidth >= picHeight:
        netInputW = NET_W
        ratio = netInputW/picWidth
        netInputH = int(ratio * picHeight)
    else:
        netInputH = NET_H
        ratio = netInputH/picHeight
        netInputW = int(ratio * picWidth)

    return netInputW, netInputH


if __name__ == '__main__':
    # getvideo('/home/chenjy531/Desktop/data/chenjy/测试视频（手势检测）/2022-12-13 17.27拍摄的影片.mov')
    # getvideo('/home/chenjy531/Desktop/data/chenjy/测试视频（手势检测）/2023-1-30 11.25拍摄的影片.mov')
    # getvideo('/home/chenjy531/Desktop/data/chenjy/测试视频（手势检测）/2022-12-13 17.31拍摄的影片.mov')
    getvideo('/home/chenjy531/Desktop/data/chenjy/测试视频（手势检测）/2022-12-16+11.35拍摄的影片.mov')
    # getvideo('/home/chenjy531/Desktop/data/chenjy/测试视频（手势检测）/IMG_0284.mov')
    # getvideo('/home/chenjy531/Desktop/data/chenjy/hand_gesture_datasets/2M-170cm/10-2M-正对摄像机吃饭.mp4')