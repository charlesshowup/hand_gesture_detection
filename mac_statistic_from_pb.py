import tensorflow._api.v2.compat.v1 as tf
from tensorflow._api.v2.compat.v1.python.framework import graph_util


def stats_graph(g):
    flops = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.float_operation(), run_meta=tf.RunMetadata(), cmd='op')
    params = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
    print(flops.total_float_ops)


def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        if "personCarDet" in pb:
            image_tensor = tf.placeholder(dtype=tf.float32,
                                          shape=[1,
                                                 128,
                                                 128,
                                                 4], name='input_placeholder')
            image_tensor2 = tf.placeholder(dtype=tf.float32,
                                          shape=[1,
                                                 128,
                                                 128,
                                                 4], name='input_std_placeholder')

            input_map = {'input_placeholder': image_tensor, 'input_std_placeholder': image_tensor2}
        else:
            image_tensor = tf.placeholder(dtype=tf.uint8,
                                          shape=[1,
                                                 288,
                                                 512,
                                                 3], name='image_tensor')

            input_map = {'image_tensor': image_tensor}
        tf.import_graph_def(graph_def, name='', input_map=input_map)

        graph = tf.get_default_graph()
        for op in graph.as_graph_def().node:
            if op.name.endswith('Conv2D'):
                print(op.name)
        return graph


# ***** (3) Load frozen graph *****
# graph = load_pb(r"./model_test_20230201.pb")
graph = load_pb(r"./model_test_20230217.pb")

print('stats after freezing')
stats_graph(graph)




