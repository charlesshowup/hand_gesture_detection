import tensorflow._api.v2.compat.v1 as tf
import numpy as np
import cv2
from src.input_pipline.input_pipeline import Pipeline
import shutil
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#tf.enable_eager_execution()
'''
labels = {
    'boxes': boxes,
    'num_boxes': num_boxes,
    'landmarks': landmarks,
    'occlude': occlude,
    'blur': blur
    'quality': quality 
}
'''

colorB = (244, 138, 21)
colorR = (26, 26, 240)
colorG = (49, 217, 57)
colorY = (32, 255, 255)
output_dir = 'test_input_pipeline_stitcher'
def main():
    #tf.enable_eager_execution()
    # # Get images and boxes

    tf.reset_default_graph()
    BATCH_SIZE = 16

    pipeline = Pipeline(
        ['/home/chenjy531/Desktop/data/trans/HAGRID_tfrecord/3d_train/shard-0000.tfrecords'],
        #['data/train_shards_celebA/shard-0000.tfrecords'],
        batch_size=BATCH_SIZE, image_size=(384, 384),
        repeat=False, shuffle=False,
        augmentation=True
    )
    features, labels = pipeline.get_batch()


    with tf.Session() as sess:
        I, B, N, L = sess.run([
            features['images'],
            labels['boxes'],
            labels['num_boxes'],
            labels['landmarks'],
            # labels['gesture_labels']
            # labels['landmark_occlude'],
            # labels['blur'],
            # labels['quality'],
        ])


    '''
    for i in range(24):
        print(B[i].shape[0])
        print(L[i].shape[0])
        print(O[i].shape[0])
        print(Bl[i].shape[0])
        print(N[i])
        print('\n')
    '''

    shutil.rmtree(output_dir, ignore_errors=True)
    os.mkdir(output_dir)

    # choose an image
    for img_idx in range(BATCH_SIZE):
        #image = np.uint8(np.transpose(I[i], [1, 2, 0])*255.0)
        image = np.uint8(I[img_idx]*255.0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        num_boxes = N[img_idx]
        boxes = B[img_idx][:num_boxes]
        landmarks = L[img_idx][:num_boxes]
        # labels = O[img_idx][:num_boxes]
        # blur = Bl[img_idx][:num_boxes]
        # quality = Q[img_idx][:num_boxes]
        img = drawBoxes(image, boxes, landmarks, labels)
        img_name = 'test%d.jpg' % img_idx
        img_path = os.path.join(output_dir, img_name)
        # cv2.imwrite(img_path, img)
        #print(img_name)
        #print(landmarks)
        #print('\n')
        cv2.imshow('face recognization', img)
        cv2.waitKey(0)
        #cv2.destroyAllWindows()

# # Show an augmented image with boxes
def drawBoxes(img, boxes, landmarks, labels):

    img_h, img_w, _ = img.shape
    y1 = boxes[:, 0] * img_h
    x1 = boxes[:, 1] * img_w
    y2 = boxes[:, 2] * img_h
    x2 = boxes[:, 3] * img_w

    #drawColor = (239, 209, 141)
    for i in range(x1.shape[0]):
        cv2.rectangle(img, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), colorB, 3)
        '''
        for j in range(5):
            if occlude[i][j] != -1.0:
                x = landmarks[i][2*j] * img_w
                y = landmarks[i][2*j+1] * img_h
                cv2.circle(img, center=(int(x), int(y)), radius=2, color=colorY, thickness=1)
        str_score = '%.2f' % quality[i]
        '''
        pos = (int(x1[i]), int(y1[i]) - 30)
        img = cv2.putText(img, str(labels[i]), pos, cv2.FONT_HERSHEY_COMPLEX, 2, colorB, 2)
    return img

if __name__ == '__main__':
    main()

