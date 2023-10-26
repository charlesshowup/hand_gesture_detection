import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import cv2
import time
import json
from tqdm import tqdm
import tensorflow._api.v2.compat.v1 as tf
from gesture_landmark_labels_detector import GestureLandmarkDetector
from pre_data.create_tfrecord import transfer_str_to_num
#from face_detector import FaceDetector
#FaceLandmarkDetector = FaceDetector
#from pose_estimation import face_orientation

colorB = (244, 138, 21)
colorR = (26, 26, 240)
colorG = (49, 217, 57)
colorY = (32, 255, 255)

# MODEL_PATH = 'model_test_20221213_1.pb'
MODEL_PATH = 'model_test_20221219.pb'


def drawBoxes(img, boxes, scores, landmarks, labels, testing):
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
        pos = (int(x1_array[i]), int(y1_array[i] + 20))
        score_string = '%.2f' % scores[i]
        if testing is True:
            gesture_string = '%.2f' % np.argmax(labels[i], axis=-1)
            img = cv2.putText(img, 'labels:{}, pro:{}'.format(gesture_string, score_string), pos, cv2.FONT_HERSHEY_COMPLEX, 1, drawColor, 1)
            #img = cv2.putText(img, score_string, pos, cv2.FONT_HERSHEY_COMPLEX, 1, drawColor, 1)
            #img = cv2.putText(img, str(quality[i]), pos, cv2.FONT_HERSHEY_COMPLEX, 0.5, drawColor, 1)
            #img = cv2.putText(img, str(i), pos, cv2.FONT_HERSHEY_COMPLEX, 1, drawColor, 1)
            for j in range(21):
                x = landmarks[i][2*j]
                y = landmarks[i][2*j+1]
                landmarks_list.append((x, y))
                cv2.circle(img, center=(int(x), int(y)), radius=3, color=colorB, thickness=-1)
        elif testing is False:
            gesture_string = int(np.argmax(labels[i], axis=-1))
            if gesture_string == 12 :
                gesture_string = 'stop'
            else:
                gesture_string = 'others'
            img = cv2.putText(img, 'labels:{}, pro:{}'.format(gesture_string, score_string), pos,
                              cv2.FONT_HERSHEY_COMPLEX, 1, drawColor, 1)
            # cv2.line(img, start_point, end_point, (0, 255, 255), 2)


    return img


def drawBoxes_only_det(img, boxes, scores):
    y1_array = boxes[:, 0]
    x1_array = boxes[:, 1]
    y2_array = boxes[:, 2]
    x2_array = boxes[:, 3]
    #img_copy = img.copy()

    drawColor = (239, 209, 141)
    for i in range(x1_array.shape[0]):
        cv2.rectangle(
            img, (int(x1_array[i]), int(y1_array[i])), (int(x2_array[i]), int(y2_array[i])), drawColor, 3)
        pos = (int(x1_array[i]), int(y1_array[i] - 20))
        score_string = '%.2f' % scores[i]
        img = cv2.putText(img, score_string, pos, cv2.FONT_HERSHEY_COMPLEX, 1, drawColor, 1)

    return img


def calc_sobel(img):
    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #if img.shape[-1] == 3:
    #    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #elif img.shape[-1] == 1:
    #    gray = img
    #else:
    #    raise Exception('img dimension error')
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        score = 0
        return score

    gray_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gray_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(gray_x)
    abs_grad_y = cv2.convertScaleAbs(gray_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    score = np.sum(grad) / np.prod(np.shape(grad))

    return score


def calc_avg(img):
    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #if img.shape[-1] == 3:
    #    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #elif img.shape[-1] == 1:
    #    gray = img
    #else:
    #    raise Exception('img dimension error')
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        score = 0
        return score

    score = np.mean(gray)

    return score


def calc_sad(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    sad = np.abs(gray-mean)
    return np.mean(sad)

def calc_ssd(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    ssd = (gray-mean)**2
    return np.mean(ssd)



def testImg(imgPath, dstPath):
    #face_landmark_detector = FaceLandmarkDetector(
    #    MODEL_PATH,
    #    gpu_memory_fraction=0.85,
    #    visible_device_list='0')
    gesture_landmark_detector = GestureLandmarkDetector(
        MODEL_PATH,
        gpu_memory_fraction=0.85)

    img = cv2.imread(imgPath)
    #img = img.copy()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    picHeight, picWidth, _ = imgRGB.shape
    netInputW, netInputH = getProperSize(picWidth, picHeight)
    imgRGB = cv2.resize(imgRGB, (netInputW, netInputH), interpolation=cv2.INTER_LINEAR)
    boxes, scores, landmarks, labels = gesture_landmark_detector(imgRGB, score_threshold=0.5)
    print(boxes)
    boxes = getOrgBoxes(boxes, picWidth / netInputW, picHeight / netInputH)
    landmarks = getOrgLandmarks(landmarks, picWidth / netInputW, picHeight / netInputH)
    print(labels)
    numBox = boxes.shape[0]
    if numBox != 0:
        img = drawBoxes(img, boxes, scores, landmarks, labels)

    # cv2.imwrite(dstPath, img)

    # img = cv2.resize(img, (512, 288))

    cv2.namedWindow("face recognization", 0)
    cv2.resizeWindow('face recognization', 800, 800)
    cv2.imshow('face recognization', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def testImg_only_det(imgPath, dstPath):
    face_detector = GestureLandmarkDetector(
        MODEL_PATH,
        gpu_memory_fraction=0.85)

    img = cv2.imread(imgPath)
    #img = img.copy()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    picHeight, picWidth, _ = imgRGB.shape
    netInputW, netInputH = getProperSize(picWidth, picHeight)
    imgRGB = cv2.resize(imgRGB, (netInputW, netInputH), interpolation=cv2.INTER_LINEAR)
    boxes, scores = face_detector(imgRGB, score_threshold=0.5)
    print(boxes)
    boxes = getOrgBoxes(boxes, picWidth / netInputW, picHeight / netInputH)
    #landmarks = getOrgLandmarks(landmarks, picWidth / netInputW, picHeight / netInputH)
    #print(quality)
    numBox = boxes.shape[0]
    if numBox != 0:
        #img = drawBoxes(img, boxes, scores, landmarks, quality, blur, occlude)
        img = drawBoxes_only_det(img, boxes, scores)

    cv2.imwrite(dstPath, img)


def time_test():
    times = []
    MODEL_PATH = 'model.pb'
    image_array = np.ones([1024, 1024, 3])
    face_landmark_detector = GestureLandmarkDetector(
        MODEL_PATH,
        gpu_memory_fraction=0.25,
        visible_device_list='0')
    for _ in range(110):
        start = time.perf_counter()
        boxes, scores, landmarks = face_landmark_detector(image_array, score_threshold=0.25)
        times.append(time.perf_counter() - start)

    times = np.array(times)
    times = times[10:]
    print(times.mean(), times.std())


def getProperSize(picWidth, picHeight):
    """
    keep ratio, long side
    :param picWidth:
    :param picHeight:
    :return:
    """
    NET_W, NET_H = 512, 512
    if picWidth >= picHeight:
        netInputW = NET_W
        ratio = netInputW/picWidth
        netInputH = int(ratio * picHeight)
    else:
        netInputH = NET_H
        ratio = netInputH/picHeight
        netInputW = int(ratio * picWidth)

    return netInputW, netInputH

def getOrgBoxes(boxes, ratioW, ratioH):
    boxes[:, 0] = boxes[:, 0] * ratioH
    boxes[:, 1] = boxes[:, 1] * ratioW
    boxes[:, 2] = boxes[:, 2] * ratioH
    boxes[:, 3] = boxes[:, 3] * ratioW
    return boxes

def getOrgLandmarks(landmarks, ratioW, ratioH):
    for j in range(21):
        landmarks[:, 2 * j] *= ratioW
        landmarks[:, 2 * j + 1] *= ratioH
    return landmarks

def prepare_img_person(image_path, net_w, net_h):
    MEANS = 0
    scale = 1
    Img = cv2.imread(image_path)
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    resized_img_w = net_w
    resized_img_h = net_h
    resized_img = cv2.resize(Img, (resized_img_w, resized_img_h))
    #input_img = np.expand_dims(resized_img, axis=0)

    input_img = resized_img.astype(np.float32) - MEANS
    input_img = input_img * scale
    return input_img


def test_dir():
    gesture_landmark_detector = GestureLandmarkDetector(
        MODEL_PATH,
        gpu_memory_fraction=0.85)
    src_dir = '/home/chenjy531/Desktop/data/chenjy/hagrid/subsample/stop'
    labels_dir = '/home/chenjy531/Desktop/data/chenjy/hagrid/ann_subsample/ann_subsample/stop.json'
    bboxes_counts = 0
    bboxes_detects = 0
    landmarks_json, bbox_json, labels_json = [], [], []
    landmarks_error, bbox_error, labels_error = [], [], []

    with open(labels_dir, 'r') as fp:
        content = json.load(fp)
    for filenames, labels in tqdm(content.items()):
        filename = filenames + '.jpg'
        img_dir = os.path.join(src_dir, filename)
        landmarks_ori = labels['landmarks']
        landmarks_json.append(np.array(landmarks_ori).reshape(-1, 1))
        bbox_ori = labels['bboxes']
        hand_gesture = labels['labels']
        hand_gesture = transfer_str_to_num(hand_gesture)
        labels_json.append(hand_gesture)
        bbox_json.append(np.array(bbox_ori))
        bboxes_counts += len(bbox_ori)
        img = cv2.imread(img_dir)
        # img = img.copy()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        picHeight, picWidth, _ = imgRGB.shape
        netInputW, netInputH = getProperSize(picWidth, picHeight)
        imgRGB = cv2.resize(imgRGB, (netInputW, netInputH), interpolation=cv2.INTER_LINEAR)
        boxes, scores, landmarks, labels = gesture_landmark_detector(imgRGB, score_threshold=0.6)
        for i in labels:
            labels_error.append(np.argmax(i))
        bboxes_detects += len(boxes)

    from sklearn.metrics import accuracy_score

    for i in labels_error:
        if i == 18:
            labels_error.remove(i)
    # original = np.reshape(labels_json, (-1, 1))
    original = np.ones(np.shape(labels_error))
    original[original == 1] = 12
    original = np.reshape(original, (-1, 1))
    print(f'original shape is {original.shape}')

    prediction = np.reshape(labels_error, (-1, 1))
    # print(prediction)
    prediction = np.array([int(i) for i in prediction])
    print(f'prediction shape is {prediction.shape}')
    accuracy = accuracy_score(original, prediction)
    error_loss_bbox = (abs(bboxes_counts - bboxes_detects) / bboxes_counts) * 100
    print('the accuracy of labels is:{:.4f}, the miss of bbox is:{:.4f}%'.format(accuracy, error_loss_bbox))






def make_dir(dirpath):
    try:
        os.mkdir(dirpath)
    except:
        pass


def test_img_dir():
    # src_dir = '/home/chenjy531/Desktop/data/chenjy/CelebA_official/Img/img_celeba.7z/img_celeba'
    src_dir = '/home/chenjy531/Desktop/data/chenjy/hagrid/subsample/palm'
    dst_dir = src_dir + '_results'
    make_dir(dst_dir)
    img_name_list = [each for each in os.listdir(src_dir) if each.endswith('.jpg')]
    for img_name in img_name_list:
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, img_name)
        testImg(src_path, dst_path)
        print(img_name)

def main():
    # main()
    imgPath = '11.jpg'
    dstPath = imgPath[:-4] + '_20210628.jpg'
    #testImg_person(imgPath, dstPath)
    #testImg(imgPath, dstPath)
    #dstPath = '11_cheek.jpg'
    testImg(imgPath, dstPath)
    #testImg_only_det(imgPath, dstPath)

    #testImg('face.jpg')
    #testImg('000388.jpg')
    #batchTest()
    #testImg('face3.jpg')
    #testImg('face8.jpg')

    #test_dir1()


if __name__ == '__main__':
    # test_img_dir()
    test_dir()
