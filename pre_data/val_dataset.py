import cv2
import json
import os
import numpy as np


def val_dataset():
    ann_dir = r'/home/chenjy531/Desktop/data/chenjy/hagrid/ann_subsample/ann_subsample'
    img_dir = r'/home/chenjy531/Desktop/data/trans/hagrid/img'
    json_list = [each for each in os.listdir(ann_dir) if each.endswith('.json')]
    json_list.sort()
    annotation_list = []
    for json_name in json_list[:1]:
        json_path = os.path.join(ann_dir, json_name)
        # print(json_name)
        with open(json_path, 'r') as fp:
            content = json.load(fp)
            for i, k in content.items():
                img_name = i + '.jpg'
                img_path = os.path.join(img_dir, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (384, 384))
                h , w, _ = img.shape
                bboxes_list = k['bboxes']
                num_box = len(bboxes_list)
                for i in range(num_box):
                    line_list = bboxes_list[i]
                    x1 = int(line_list[0]) * w
                    y1 = int(line_list[1]) * h
                    x2 = (int(line_list[2]) * w) + x1
                    y2 = (int(line_list[3]) * h) + y1
                    print(y2)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255))

                landmark_list = k['landmarks']
                num_landmarks = len(landmark_list)
                for i in range(num_landmarks):
                    landmark = np.squeeze(landmark_list[i])
                    if landmark.shape == (0, ):
                        break
                    else:
                        # print(landmark.shape)
                        landmark = np.reshape(landmark, (21, 2))
                        print(landmark)
                        print(landmark.shape)
                        num_landmarks_single = len(landmark)
                        for m in range(num_landmarks_single):
                            x = int(landmark[m][0] * w)
                            y = int(landmark[m][1] * h)
                            # print(x)


                            cv2.circle(img, (x, y), radius=1, color=(0, 0, 255), thickness=-1)

                cv2.imshow('', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == '__main__':
    val_dataset()

