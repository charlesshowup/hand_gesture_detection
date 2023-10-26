from config import params
from src.detector import Detector
from src.anchor_generator import AnchorGenerator, ANCHOR_SPECIFICATIONS
from src.network import FeatureExtractor
from src.input_pipline.input_pipeline import Pipeline
from loss import add_weight_decay, get_multi_loss
import os
import tensorflow._api.v2.compat.v1 as tf
tf.disable_eager_execution()
#import time
from pre_data.out_img import draw_pic
from datetime import datetime
import numpy as np
import cv2
import copy

def train_loop():
    os.environ['CUDA_VISIBLE_DEVICES'] = params['used_gpus']
    model_params = params['model_params']
    input_params = params['input_pipeline_params']

    is_training = True
    # anchor_generator = anchor_generator
    features, labels = get_input(is_training=True, is_aug=True)
    '''
    while True:
        imges, label = tf.Session().run([features, labels])
        img = imges.get('images')
        box_all = label.get('boxes')
        landmarks = label.get('landmarks')
        filenames = imges.get('filenames')
        gesture_labels = label.get('gesture_labels')
        number_img = len(img)
        # print(gesture_labels)
        # print(label)
        # exit(0)
        # print('dd', box_all[2])
        for num in range(number_img):
            # print(f"n: {num}")
            img_u8 = np.array(np.clip(img[num] * 256, 0, 255), dtype=np.uint8)
            # cv2.imshow("", img_u8)
            # cv2.waitKey(0)
            h, w, _ = img_u8.shape
            # print(f'h:{h}, w:{w}')
            name = filenames[num]
            box = np.array(box_all[num])
            # print(box.shape)
            img_u8 = img_u8[:, :, ::-1]
            img_u8 = copy.deepcopy(img_u8)
            for i in range(box.shape[0]):
                y1 = int(box[i][0] * 384)
                x1 = int(box[i][1] * 384)
                y2 = int(box[i][2] * 384)
                x2 = int(box[i][3] * 384)
                cv2.rectangle(img_u8, (x1, y1), (x2, y2), (255, 255, 255), )
                cv2.putText(img_u8, '{}:{}'.format('gesture_label', 'ddd'), (x1, y1),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            0.5, (0, 255, 255), 1)
            keynode = landmarks[num]
            # print(keynode)
            if len(keynode) == 0:
                pass
            else:
                keynode = np.reshape(keynode, (-1, 2))
                # print(keynode)
                # print(keynode.shape)
                for i in range(len(keynode)):
                    x = int(keynode[i][0] * 384)
                    y = int(keynode[i][1] * 384)
                    cv2.circle(img_u8, center=(x, y), radius=2, color=(0, 0, 255), thickness=-1)
                # print(keynode.shape)
                # print(keynode)
                # draw_pic(img_u8, keynode)
            # cv2.imwrite('/home/chenjy531/Desktop/data/trans/hagrid/no_augmentation/' + str(name), img_u8)
            cv2.imshow("", img_u8)
            cv2.waitKey(0)


    exit(0)
    '''
    feature_extractor = FeatureExtractor(is_training)
    anchor_generator = AnchorGenerator(ANCHOR_SPECIFICATIONS)
    detector = Detector(features['images'], feature_extractor, anchor_generator)
    with tf.name_scope('weight_decay'):
        add_weight_decay(model_params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()
    losses = detector.loss(labels, model_params)

    if not model_params['is_fine_tune_landmark']:
        if model_params['use_multi_loss']:
            if input_params['use_bbox_only']:
                loss_dict = {
                    'localization_loss': losses['localization_loss'],
                    'classification_loss': losses['classification_loss'],
                }
            else:
                loss_dict = {
                    'localization_loss': losses['localization_loss'],
                    'landmark_loss': losses['landmark_loss'],
                    'classification_loss': losses['classification_loss'],
                    # 'label_loss': losses['label_loss'],
                    # 'quality_loss': losses['quality_loss'],
                    # losses['blur_loss']
                }
            loss_sum = get_multi_loss(loss_dict)
            tf.losses.add_loss(loss_sum)
        else:
            if input_params['use_bbox_only']:
                tf.losses.add_loss(model_params['localization_loss_weight'] * losses['localization_loss'])
                tf.losses.add_loss(model_params['classification_loss_weight'] * losses['classification_loss'])
            else:
                tf.losses.add_loss(model_params['localization_loss_weight'] * losses['localization_loss'])
                tf.losses.add_loss(model_params['landmark_loss_weight'] * losses['landmark_loss'])
                # tf.losses.add_loss(model_params['label_loss_weight'] * losses['label_loss'])
                # tf.losses.add_loss(model_params['classification_loss_weight'] * losses['classification_loss'])
                # tf.losses.add_loss(model_params['quality_loss_weight'] * losses['quality_loss'])
                # tf.losses.add_loss(model_params['blur_loss_weight'] * losses['blur_loss'])
                # tf.losses.add_loss(model_params['occlude_loss_weight'] * losses['occlude_loss'])
    else:
        tf.losses.add_loss(losses['landmark_loss'])

    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('localization_loss', losses['localization_loss'])
    tf.summary.scalar('landmark_loss', losses['landmark_loss'])
    tf.summary.scalar('classification_loss', losses['classification_loss'])
    # tf.summary.scalar('label_loss', losses['label_loss'])
    # tf.summary.scalar('quality_loss', losses['quality_loss'])
    # tf.summary.scalar('blur_loss', losses['blur_loss'])
    # tf.summary.scalar('occlude_loss', losses['occlude_loss'])


    if not model_params['is_fine_tune_landmark']:
        total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
    else:
        total_loss = tf.losses.get_total_loss(add_regularization_losses=False)
    tf.summary.scalar('total_loss', total_loss)

    with tf.variable_scope('learning_rate'):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                      trainable=False)
        if model_params['use_cosine_decay'] == True:
            from learning_schedues import cosine_decay_with_warmup
            learning_rate = cosine_decay_with_warmup(global_step=global_step,
                                                     learning_rate_base=model_params['learning_rate_base'],
                                                     total_steps=model_params['total_steps'],
                                                     warmup_learning_rate=model_params['warmup_learning_rate'],
                                                     warmup_steps=model_params['warmup_steps'],
                                                     hold_base_rate_steps=model_params['hold_base_rate_steps'])
        else:
            boundaries = [float(b) for b in model_params['lr_boundaries']]
            values = [float(v) for v in model_params['lr_values']]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    sess = tf.Session(config=get_sess_config())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    start_time = datetime.now()
    training_steps = input_params['num_steps']

    saver = restore_fake_params_from_checkpoint(sess)

    if params['save_summary']:
       summary_writer = tf.summary.FileWriter(model_params['model_dir'], sess.graph)
       merged_summary = tf.summary.merge_all()

    every_n_steps_print_info = 100
    every_n_steps_save_ckpt = params['save_ckpt_every_n_steps']
    for step in range(training_steps):
        _, loss_value, learn_rate = sess.run([train_op, total_loss, learning_rate])
        if params['save_summary']:
            train_summary = sess.run(merged_summary)
            summary_writer.add_summary(train_summary, step)
        if step % every_n_steps_print_info == 0:
            duration = datetime.now() - start_time
            start_time = datetime.now()
            print('step = %d, loss = %.2f, learning_rate = %.6f (%.2f sec)' % (step, loss_value, learn_rate, duration.total_seconds()))

        if (step % every_n_steps_save_ckpt == 0) or (step + 1) == training_steps:
            checkpoint_path = os.path.join(model_params['model_dir'], 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)

    coord.request_stop()
    coord.join(threads)
    if params['save_summary']:
        summary_writer.close()
    print('finish time: %s' % datetime.now())


def restore_fake_params_from_checkpoint(sess):
    if params['is_train_from_begining']:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
    else:
        g_list = tf.global_variables()
        train_var_list = tf.trainable_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        quan_moving_vars = [g for g in g_list if 'moving_maxValue' in g.name]
        quan_moving_vars += [g for g in g_list if 'moving_minValue' in g.name]
        quan_moving_vars += [g for g in g_list if 'moving_values' in g.name]
        quan_debug_vars = [g for g in g_list if 'max_out_channels' in g.name]
        train_var_list += bn_moving_vars
        train_var_list_quantize = train_var_list + quan_moving_vars + quan_debug_vars
        if params['quantization_params']['is_train_fake_model']:
            # restore parameters excluding quantization parameters
            if params['quantization_params']['is_restore_from_float_ckpt']:
                train_var_list_temp = train_var_list_quantize
                train_var_list_restore = [g for g in train_var_list_temp if
                                          'landmark_encoding_predictor_4/quan_outlayer/moving_maxValue' not in g.name]
                train_var_list_restore = [g for g in train_var_list_restore if
                                          'landmark_encoding_predictor_4/quan_outlayer/moving_minValue' not in g.name]
                train_var_list_restore = [g for g in train_var_list_restore if
                                          'landmark_encoding_predictor_3/quan_outlayer/moving_maxValue' not in g.name]
                train_var_list_restore = [g for g in train_var_list_restore if
                                          'landmark_encoding_predictor_3/quan_outlayer/moving_minValue' not in g.name]
                train_var_list_restore = [g for g in train_var_list_restore if
                                          'landmark_encoding_predictor_2/quan_outlayer/moving_maxValue' not in g.name]
                train_var_list_restore = [g for g in train_var_list_restore if
                                          'landmark_encoding_predictor_2/quan_outlayer/moving_minValue' not in g.name]
                train_var_list_restore = [g for g in train_var_list_restore if
                                          'landmark_encoding_predictor_1/quan_outlayer/moving_maxValue' not in g.name]
                train_var_list_restore = [g for g in train_var_list_restore if
                                          'landmark_encoding_predictor_1/quan_outlayer/moving_minValue' not in g.name]
                train_var_list_restore = [g for g in train_var_list_restore if
                                          'landmark_encoding_predictor_0/quan_outlayer/moving_maxValue' not in g.name]
                train_var_list_restore = [g for g in train_var_list_restore if
                                          'landmark_encoding_predictor_0/quan_outlayer/moving_minValue' not in g.name]

            else:
                train_var_list_restore = train_var_list_quantize
            saver = tf.train.Saver(train_var_list_restore)
            saver.restore(sess, tf.train.latest_checkpoint(params['model_params']['model_dir']))
            exclude = [val for val in g_list if val not in train_var_list_restore]
            # print(exclude)
            sess.run(tf.variables_initializer(exclude))
            sess.run(tf.local_variables_initializer())
            # save quantization parameters when save
            saver = tf.train.Saver(train_var_list_quantize)
        else:
            sess.run(tf.variables_initializer(g_list))
            saver = tf.train.Saver(train_var_list_quantize)
    return saver


def get_sess_config():
    config = tf.ConfigProto()
    #config.gpu_options.visible_device_list = GPU_TO_USE
    config.gpu_options.allow_growth = True
    return config



def get_input(is_training=True, is_aug=True):
    input_params = params['input_pipeline_params']
    image_size = input_params['image_size'] if is_training else None
    # (for evaluation i use images of different sizes)
    dataset_path = input_params['train_dataset'] if is_training else input_params['val_dataset']
    batch_size = input_params['batch_size'] if is_training else 1
    # for evaluation it's important to set batch_size to 1

    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    def input_fn():
        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            pipeline = Pipeline(
                filenames,
                batch_size=batch_size, image_size=image_size,
                repeat=is_training, shuffle=is_training,
                # repeat=is_training, shuffle=is_training,
                augmentation=is_aug
            )
            features, labels = pipeline.get_batch()
        return features, labels

    return input_fn()


if __name__ == '__main__':
    train_loop()

