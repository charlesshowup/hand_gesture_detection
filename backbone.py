import tensorflow._api.v2.compat.v1 as tf
import tf_slim as slim

import numpy as np

# modify to vovnet paper backbone

NUM_CENTERS = [[4, 4], [2, 2], [1, 1], [1, 1], [1, 1]]
RATIOS_NUM = [1, 1, 1, 1, 1, 1, 1]

weight_decay = False
TRAINABLE = True
BN_TRAINING = False


def create_variables(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var


def variable_with_weight_decay(name, shape, wd):
    var = create_variables(
      name,
      shape,
      tf.glorot_uniform_initializer())
      #tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
      # tf.glorot_uniform_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def batch_normalization_layer(input_layer,  train_phase=True):
    bn_layer = tf.layers.batch_normalization(input_layer, training=train_phase, name='bn')
    return bn_layer


def batch_norm(x):
    x = tf.layers.batch_normalization(
        x,
        training=BN_TRAINING,
        trainable=TRAINABLE,
        name='bn'
    )
    return x

def conv_layer(input_layer, filter_shape, stride, padding, wd=None):

    #filter_kernel = variable_with_weight_decay(name='weights', shape=filter_shape, wd=wd)
    #conv_layer = tf.nn.conv2d(input_layer, filter_kernel, strides=[1, stride, stride, 1], padding=padding, trainable=TRAINABLE)
    #biases = create_variables('biases', filter_shape[-1], tf.constant_initializer(0.0))
    #pre_activation = tf.nn.bias_add(conv_layer, biases)

    kernel_h, kernel_w, in_channels, out_channels = filter_shape
    output = slim.conv2d(input_layer, out_channels, (kernel_h, kernel_w),
                         stride=stride,
                         padding=padding,
                         trainable=TRAINABLE,
                         activation_fn=None,
                         normalizer_fn=None,
                         data_format='NHWC'
                         ),
    return output


def depthwise_layer(input_layer, filter_shape, stride, padding, wd=None):

    filter_kernel = variable_with_weight_decay(name='weights', shape=filter_shape, wd=wd)
    conv_layer = tf.nn.depthwise_conv2d(input_layer, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    biases = create_variables('biases', filter_shape[-2], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv_layer, biases)
    return pre_activation


def depthwise_bn_relu_layer(input_layer, ks, stride, padding, bn=True, wd=None):
    input_channel = input_layer.get_shape().as_list()[-1]
    filter_kernel = variable_with_weight_decay(name='weights', shape=[ks, ks, input_channel, 1], wd=wd)
    conv_layer = tf.nn.depthwise_conv2d(input_layer, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    bn_layer = batch_normalization_layer(conv_layer, train_phase=bn)
    output = tf.nn.relu(bn_layer)
    return output


def conv_bn_relu_layer(input_layer, filter_shape, stride, padding, bn=True, wd=None):

    #filter_kernel = variable_with_weight_decay(name='weights', shape=filter_shape, wd=wd)
    #conv_layer = tf.nn.conv2d(input_layer, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    #bn_layer = batch_normalization_layer(conv_layer, train_phase=bn)
    #output = tf.nn.relu(bn_layer)
    kernel_h, kernel_w, in_channels, out_channels = filter_shape
    output = slim.conv2d(input_layer, out_channels, (kernel_h, kernel_w),
                         stride=stride,
                         padding=padding,
                         trainable=TRAINABLE,
                         activation_fn=tf.nn.relu,
                         normalizer_fn=batch_norm,
                         data_format='NHWC'
                         )
    return output


def conv_active_layer(input_layer, filter_shape, stride, padding, bais_initializer_value=0.0, wd=None,
                      if_active=True, if_relu=True):

    filter_kernel = variable_with_weight_decay(name='weights', shape=filter_shape, wd=wd)
    conv_layer = tf.nn.conv2d(input_layer, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    biases = create_variables('biases', filter_shape[-1], tf.constant_initializer(bais_initializer_value))
    pre_activation = tf.nn.bias_add(conv_layer, biases)
    if if_active:
        if if_relu:
            pre_activation = tf.nn.relu(pre_activation)
        else:
            pre_activation = tf.nn.sigmoid(pre_activation)

    return pre_activation


def steam_block(input_layer, bn=True, wd=None):
    with tf.variable_scope('conv1'):
        conv1_layer = conv_bn_relu_layer(input_layer, [3, 3, 3, 16], stride=2, padding='SAME', bn=bn, wd=wd)
    with tf.variable_scope('conv2'):
        conv2_layer = conv_bn_relu_layer(conv1_layer, [3, 3, 16, 16], stride=1, padding='SAME', bn=bn, wd=wd)
    with tf.variable_scope('conv3'):
        conv3_layer = conv_bn_relu_layer(conv2_layer, [3, 3, 16, 32], stride=2, padding='SAME', bn=bn, wd=wd)

    #maxpool_layer = transition_layer(conv1_1_layer, 16, is_pool=True, bn=bn, wd=weight_decay)

    return conv3_layer


def dense_layer(input_layer, k, bottleneck_width, bn=True, wd=None):
    input_channel = input_layer.get_shape().as_list()[-1]
    bottle_channel = k * bottleneck_width

    with tf.variable_scope('left_conv1'):
        left_conv1_layer = conv_bn_relu_layer(input_layer, [1, 1, input_channel, bottle_channel], 1,
                                              padding='SAME', bn=bn, wd=wd)

    with tf.variable_scope('left_conv2'):
        left_conv2_layer = conv_bn_relu_layer(left_conv1_layer, [3, 3, bottle_channel, k], 1,
                                              padding='SAME', bn=bn, wd=wd)

    output = tf.concat([input_layer, left_conv2_layer], axis=3)
    return output


def dense_layer2(input_layer, k=16, bottle=1, ext=False, bn=True, wd=None):
    input_channel = input_layer.get_shape().as_list()[-1]
    bottle_channel = k * bottle

    with tf.variable_scope('linear_conv1'):
        conv1_layer = conv_bn_relu_layer(input_layer, [1, 1, input_channel, bottle_channel], 1,
                                         padding='SAME', bn=bn, wd=wd)

    with tf.variable_scope('conv2'):
        conv2_layer = conv_bn_relu_layer(conv1_layer, [3, 3, bottle_channel,  k], 1,
                                         padding='SAME', bn=bn, wd=wd)
    if ext:
        with tf.variable_scope('linear_conv2'):
            conv2_layer = conv_bn_relu_layer(conv2_layer, [1, 1, k, bottle_channel], 1,
                                             padding='SAME', bn=bn, wd=wd)

    output = tf.concat([input_layer, conv2_layer], axis=-1)
    return output


def dense_block(input_layer, num_dense_layer, k, bottleneck_width, bn=True, wd=None):
    output = input_layer
    for i in range(num_dense_layer):
        with tf.variable_scope('dense_layer_%d' % (i + 1)):
            output = dense_layer(output, k, bottleneck_width, bn=bn, wd=wd)

    return output


def dense_block2(input_layer, num_dense_layer, k, bottleneck_width, ext=False, bn=True, wd=None):
    output = input_layer
    for i in range(num_dense_layer):
        with tf.variable_scope('dense_layer_%d' % (i + 1)):
            output = dense_layer2(output, k=k, bottle=bottleneck_width, ext=ext, bn=bn, wd=wd)

    return output


def transition_layer(input_layer, output_channel, is_pool=True, bn=True, wd=None):
    input_channel = input_layer.get_shape().as_list()[-1]

    with tf.variable_scope('conv1'):
        output = conv_bn_relu_layer(input_layer, [1, 1, input_channel, output_channel], 1,
                                    padding='SAME', bn=bn, wd=wd)

    if is_pool:
        output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return output


def conv_bn_layer(input_layer, filter_shape, stride, padding, bn=True, wd=None):

    #filter_kernel = variable_with_weight_decay(name='weights', shape=filter_shape, wd=wd)
    #conv_layer = tf.nn.conv2d(input_layer, filter_kernel, strides=[1, stride, stride, 1], padding=padding)
    #bn_layer = batch_normalization_layer(conv_layer, train_phase=bn)
    kernel_h, kernel_w, in_channels, out_channels = filter_shape
    output = slim.conv2d(input_layer, out_channels, (kernel_h, kernel_w),
                         stride=stride,
                         padding=padding,
                         trainable=TRAINABLE,
                         activation_fn=None,
                         normalizer_fn=batch_norm,
                         data_format='NHWC'
                         )
    return output


def res_block(input_layer, bn=True, wd=None):
    input_channel = input_layer.get_shape().as_list()[-1]
    with tf.variable_scope('left_conv1'):
        left_conv1_layer = conv_bn_relu_layer(input_layer, [1, 1, input_channel, 128], 1,
                                              padding='SAME', bn=bn, wd=wd)

    with tf.variable_scope('left_conv2'):
        left_conv2_layer = conv_bn_layer(left_conv1_layer, [3, 3, 128, 128], 1,
                                              padding='SAME', bn=bn, wd=wd)


    with tf.variable_scope('right_conv1'):
        right_conv1_layer = conv_bn_layer(input_layer, [1, 1, input_channel, 128], 1,
                                               padding='SAME', bn=bn, wd=wd)

    output_add = left_conv2_layer + right_conv1_layer
    output = tf.nn.relu(output_add)
    return output


def dense_block_ext(input_layer, bn=True, wd=None):
    input_channel = input_layer.get_shape().as_list()[-1]
    with tf.variable_scope('left_conv1'):
        left_conv1_layer = conv_bn_relu_layer(input_layer, [1, 1, input_channel, 192], 1,
                                              padding='SAME', bn=bn, wd=wd)

    with tf.variable_scope('left_conv2'):
        left_conv2_layer = conv_bn_relu_layer(left_conv1_layer, [3, 3, 192, 192], 2,
                                              padding='SAME', bn=bn, wd=wd)

    with tf.variable_scope('right_conv1'):
        pool_layer = tf.nn.max_pool(input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME')
        right_conv1_layer = conv_bn_relu_layer(pool_layer, [1, 1, input_channel, 192], 1,
                                               padding='SAME', bn=bn, wd=wd)

    output = tf.concat([left_conv2_layer, right_conv1_layer], axis=-1)
    return output


def dense_block_ext2(input_layer, bn=True, wd=None):
    input_channel = input_layer.get_shape().as_list()[-1]
    with tf.variable_scope('left_conv1'):
        left_conv1_layer = conv_bn_relu_layer(input_layer, [1, 1, input_channel, 128], 1,
                                              padding='SAME', bn=bn, wd=wd)

    with tf.variable_scope('left_conv2'):
        left_conv2_layer = conv_bn_relu_layer(left_conv1_layer, [3, 3, 128, 128], 2,
                                              padding='SAME', bn=bn, wd=wd)

    with tf.variable_scope('right_conv1'):
        pool_layer = tf.nn.max_pool(input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                      padding='SAME')
        right_conv1_layer = conv_bn_relu_layer(pool_layer, [1, 1, input_channel, 128], 1,
                                               padding='SAME', bn=bn, wd=wd)

    output = tf.concat([left_conv2_layer, right_conv1_layer], axis=-1)
    return output


def _osa_module(input_layer, num_layers, layer_c_out):
    concat_list = [input_layer]
    for i in range(num_layers):
        with tf.variable_scope('layer_%d' % (i + 1)):
            input_channel = input_layer.shape[-1].value
            input_layer = \
                conv_bn_relu_layer(
                    input_layer,
                    filter_shape=[3, 3, input_channel, layer_c_out],
                    stride=1, padding='SAME')
            concat_list.append(input_layer)
    return tf.concat(concat_list, axis=-1)


def inference(input_tensor_batch, bn, trainable, reuse=False):
    global TRAINABLE
    TRAINABLE = trainable
    global BN_TRAINING
    BN_TRAINING = bn
    with tf.variable_scope('base_vovnet', reuse=reuse):
        #layers = []
        with tf.variable_scope('steam_block', reuse=reuse):  # downsample
            output_layer = steam_block(input_tensor_batch, bn=bn, wd=weight_decay)

        with tf.variable_scope('stage_1', reuse=reuse):
            with tf.variable_scope('dense_block'):
                output_layer = _osa_module(output_layer, num_layers=5, layer_c_out=32)
            with tf.variable_scope('transition_layer'):  # downsample
                output_layer = transition_layer(output_layer, 64, is_pool=True, bn=bn, wd=weight_decay)

        with tf.variable_scope('stage_2', reuse=reuse):
            with tf.variable_scope('dense_block'):
                output_layer = _osa_module(output_layer, num_layers=5, layer_c_out=40)
                pre_branch1 = output_layer

            with tf.variable_scope('transition_layer'): # downsample
                output_layer = transition_layer(output_layer, 128, is_pool=True, bn=bn, wd=weight_decay)

        with tf.variable_scope('stage_3', reuse=reuse):
            with tf.variable_scope('dense_block_0'):
                output_layer = _osa_module(output_layer, num_layers=5, layer_c_out=48)

            with tf.variable_scope('transition_layer_0'): # downsample
                output_layer = transition_layer(output_layer, 192, is_pool=False, bn=bn, wd=weight_decay)

        with tf.variable_scope('stage_4', reuse=reuse):
            with tf.variable_scope('dense_block'):
                pre_feature = _osa_module(output_layer, num_layers=5, layer_c_out=56)

        with tf.variable_scope('stage_6', reuse=reuse):
            with tf.variable_scope('transition_layer0'):
                output_layer = transition_layer(pre_branch1, 128, is_pool=False, bn=bn, wd=weight_decay)

            with tf.variable_scope('dense_block0'):
                output_layer = _osa_module(output_layer, num_layers=4, layer_c_out=32)

            with tf.variable_scope('transition_layer1'):
                output_layer = transition_layer(output_layer, 128, is_pool=False, bn=bn, wd=weight_decay)

            with tf.variable_scope('dense_block1'):
                output_layer = _osa_module(output_layer, num_layers=4, layer_c_out=32)

            with tf.variable_scope('transition_layer2'):
                output_layer = transition_layer(output_layer, 128, is_pool=False, bn=bn, wd=weight_decay)

            with tf.variable_scope('dense_block2'):
                feature1 = _osa_module(output_layer, num_layers=4, layer_c_out=32)

        with tf.variable_scope('stage_7', reuse=reuse):
            with tf.variable_scope('transition_layer0'):
                output_layer = transition_layer(pre_feature, 128, is_pool=False, bn=bn, wd=weight_decay)

            with tf.variable_scope('dense_block0'):
                #feature2 = dense_block2(output_layer, 4, k=32, bottleneck_width=1, bn=bn, wd=weight_decay)
                feature2 = _osa_module(output_layer, num_layers=4, layer_c_out=32)

        with tf.variable_scope('stage_8', reuse=reuse):
            with tf.variable_scope('transition_layer0'):
                # output_channel = output_layer.get_shape().as_list()[-1]
                output_layer = transition_layer(pre_feature, 128, is_pool=False, bn=bn,
                                                wd=weight_decay)

            with tf.variable_scope('dense_block0'):
                #feature3 = dense_block2(output_layer, 4, k=32, bottleneck_width=1, bn=bn, wd=weight_decay)
                feature3 = _osa_module(output_layer, num_layers=4, layer_c_out=32)

        with tf.variable_scope('stage_9', reuse=reuse):  # downsample 32
            with tf.variable_scope('transition_layer0'):
                output_layer = transition_layer(pre_feature, 128, is_pool=True, bn=bn,
                                                wd=weight_decay)

            with tf.variable_scope('dense_block0'):
                feature4 = _osa_module(output_layer, num_layers=4, layer_c_out=32)

        with tf.variable_scope('stage_10', reuse=reuse):
            with tf.variable_scope('transition_layer0'): # j
                output_layer = transition_layer(pre_feature, 128, is_pool=True, bn=bn, wd=weight_decay)

            with tf.variable_scope('dense_block0'):
                feature5 = _osa_module(output_layer, num_layers=4, layer_c_out=32)

            #with tf.variable_scope('transition_layer1'):
            #    output_layer = transition_layer(output_layer, 128, is_pool=True, bn=bn,
            #                                    wd=weight_decay)

            #with tf.variable_scope('dense_block1'):
            #    feature5 = _osa_module(output_layer, num_layers=4, layer_c_out=32)

        return [feature1, feature2, feature3, feature4, feature5]


def upsample(input_layer):
    with tf.variable_scope('up_sample'):
        with tf.variable_scope('conv1'):
            upsample_channel = 128
            input_channel = input_layer.shape[-1].value
            output_layer = \
                conv_bn_relu_layer(
                    input_layer,
                    [1, 1, input_channel, upsample_channel*4],
                    stride=1, padding='SAME')

        split_num_output_list = [upsample_channel] * 4
        splits = tf.split(output_layer, split_num_output_list, axis=-1)
        fm_list = []
        for i, fm in enumerate(splits):
            with tf.variable_scope('conv%d' % (i+2)):
                fm_out = \
                    conv_bn_relu_layer(fm, [1, 1, upsample_channel, upsample_channel],
                                       stride=1, padding='SAME')
            fm_list.append(fm_out)
        output = tf.concat(fm_list, axis=-1)
        output = tf.nn.depth_to_space(output, block_size=2)
        with tf.variable_scope('conv6'):
            output_layer = \
                conv_bn_relu_layer(
                    output, [3, 3, upsample_channel, upsample_channel],
                    stride=1, padding='SAME')
        return output_layer



def create_class_head(feature, head_idx, num_predictions_per_location, trainable):
    biases = np.zeros(
        [num_predictions_per_location], dtype='float32')
    biases[:] = np.log(0.01)  # object class
    #biases = biases.reshape(num_predictions_per_location * 2)
    kernel_size = 3

    y = slim.conv2d(
        feature, num_predictions_per_location,
        [kernel_size, kernel_size], activation_fn=None, scope='class_predictor_%d' % head_idx,
        data_format='NHWC', padding='SAME',
        normalizer_fn=batch_norm,
        biases_initializer=tf.constant_initializer(biases),
        trainable=trainable
    )
    return y


def create_box_head(feature, head_idx, num_predictions_per_location, trainable):
    kernel_size = 3
    y = slim.conv2d(
        feature, num_predictions_per_location * 4,
        [kernel_size, kernel_size], activation_fn=None, scope='box_encoding_predictor_%d' % head_idx,
        normalizer_fn=batch_norm,
        data_format='NHWC', padding='SAME',
        trainable=trainable
    )
    return y


def create_landmark_head(feature, head_idx, num_predictions_per_location, trainable):
    kernel_size = 3
    y = slim.conv2d(
        feature, num_predictions_per_location * 42,
        [kernel_size, kernel_size], activation_fn=None, scope='landmark_encoding_predictor_%d' % head_idx,
        normalizer_fn=batch_norm,
        data_format='NHWC', padding='SAME',
        trainable=trainable
    )
    return y



def create_occlude_head(feature, head_idx, num_predictions_per_location, trainable):
    biases = np.zeros(
        [num_predictions_per_location * 5], dtype='float32')
    biases[:] = np.log(0.5)
    kernel_size = 3

    y = slim.conv2d(
        feature, num_predictions_per_location * 5,
        [kernel_size, kernel_size], activation_fn=None, scope='occlude_predictor_%d' % head_idx,
        normalizer_fn=batch_norm,
        data_format='NHWC', padding='SAME',
        biases_initializer=tf.constant_initializer(biases),
        trainable=trainable
    )
    return y


def create_label_head(feature, head_idx, num_predictions_per_location, trainable):
    biases = np.zeros([num_predictions_per_location*19], dtype='float32')
    biases[:] = np.log(0.1)
    kernel_size = 3

    y = slim.conv2d(
        feature, num_predictions_per_location * 19,
        [kernel_size, kernel_size], activation_fn=None, scope='label_predictor_%d' % head_idx,
        normalizer_fn=batch_norm,
        data_format='NHWC', padding='SAME',
        biases_initializer=tf.constant_initializer(biases),
        trainable=trainable
    )
    return y


def create_quality_head(feature, head_idx, num_predictions_per_location, trainable):
    biases = np.zeros([num_predictions_per_location], dtype='float32')
    biases[:] = np.log(0.1)
    kernel_size = 3
    # biases = biases.reshape(num_predictions_per_location)

    y = slim.conv2d(
        feature, num_predictions_per_location,
        [kernel_size, kernel_size], activation_fn=None, scope='quality_predictor_%d' % head_idx,
        normalizer_fn=batch_norm,
        data_format='NHWC', padding='SAME',
        biases_initializer=tf.constant_initializer(biases),
        trainable=trainable
    )
    return y


def create_blur_head(feature, head_idx, num_predictions_per_location, trainable):
    biases = np.zeros([num_predictions_per_location], dtype='float32')
    biases[:] = np.log(0.1)
    kernel_size = 3
    # biases = biases.reshape(num_predictions_per_location)

    y = slim.conv2d(
        feature, num_predictions_per_location,
        [kernel_size, kernel_size], activation_fn=None, scope='blur_predictor_%d' % head_idx,
        normalizer_fn=batch_norm,
        data_format='NHWC', padding='SAME',
        biases_initializer=tf.constant_initializer(biases),
        trainable=trainable
    )
    return y







