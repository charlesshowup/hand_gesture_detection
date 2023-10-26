from tensorflow._api.v2.compat.v1.compat.v1.python import pywrap_tensorflow._api.v2.compat.v1
import os
import numpy as np
from backbone import inference
import tensorflow._api.v2.compat.v1.compat.v1 as tf
from tensorflow._api.v2.compat.v1.compat.v1.python.framework import graph_util


def stas_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('Flops:{}, TRAINABLE_PARAMs:{}'.format(flops.total_float_ops, params.total_parameters))


def load_pd(pd):
    with tf.gfile.GFile(pd, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph, name='')
        return graph


with tf.Graph().as_default() as graph:
    A = tf.placeholder(dtype=tf.float32, shape=[None, 288, 512, 3])
    output = inference(A, False, False)

    print('stats before freezing')
    stas_graph(graph)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output_graph = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['stage_10'])
        with tf.gfile.GFile('graph.pd', 'wb') as f:
            f.write(output_graph.SerializerToString())

graph = load_pd('./model_test_20230110.pb')
print('stats after freezing')
stas_graph(graph)




'''
model_path = '../../checkpoint_train_test/model.ckpt-736001'
# print(os.listdir(model_path))
reader = pywrap_tensorflow._api.v2.compat.v1.NewCheckpointReader(model_path)
var_to_shape_map = reader.get_variable_to_shape_map()
total_parameters = 0
for key in var_to_shape_map:
    shape = np.shape(reader.get_tensor(key))    
    shape = list(shape)
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim

    total_parameters += variable_parameters

print(total_parameters)
'''
