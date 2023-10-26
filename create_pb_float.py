import tensorflow._api.v2.compat.v1 as tf
from src.detector import Detector
from src.anchor_generator import AnchorGenerator, ANCHOR_SPECIFICATIONS
from src.network import FeatureExtractor
from tensorflow._api.v2.compat.v1.python.tools import freeze_graph
import argparse
import os
from config import params
model_params = params['model_params']
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

"""Create a .pb frozen inference graph from a SavedModel."""

INTELIF_TEST = True


def main():

    #output_pb_path = 'model_only_det_20210322.pb'
    #output_pb_path = '/data/liuwt/faceDet_15FPS_20210318_V4.2/model_intellif_20210603.pb'
    #output_pb_path = '/data/liuwt/faceDet_15FPS_20210318_V4.2/model_quality_20210628.pb'
    #output_pb_path = '/data/liuwt/faceDet_15FPS_20210318_V4.2/model_darkface_20210701.pb'
    #output_pb_path = '/data/liuwt/faceDet_15FPS_20210318_V4.2/model_test_interlif_20210816.pb'
    output_pb_path = 'model_test_20230222.pb'

    height = None
    width = None
    images = tf.placeholder(dtype=tf.uint8, shape=[None, height, width, 3], name='image_tensor')
    features = {'images': tf.to_float(images)*(1.0/255.0)}
    feature_extractor = FeatureExtractor(is_training=False)
    anchor_generator = AnchorGenerator(ANCHOR_SPECIFICATIONS)
    detector = Detector(features['images'], feature_extractor, anchor_generator)
    if INTELIF_TEST:
        model_params['score_threshold'] = 0.5
    predictions = detector.get_predictions(
        score_threshold=model_params['score_threshold'],
        iou_threshold=model_params['iou_threshold'],
        max_boxes=model_params['max_boxes']
    )
    for name, tensor in predictions.items():
        tf.identity(tensor, name=name)


    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    sess = tf.Session(graph=graph, config=config)
    saver = tf.train.Saver()
    #tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], ARGS.saved_model_folder)

    #for n in graph.as_graph_def().node:
    #    if 'boxes' in n.name:
    #        print(n.name)

    # output ops
    #if INTELIF_TEST:
    #    saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_train_darkface'))
    saver.restore(sess, tf.train.latest_checkpoint(model_params['model_dir']))
    keep_nodes = ['boxes', 'scores', 'num_boxes', 'landmarks']

    input_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(),
        output_node_names=keep_nodes
    )
    output_graph_def = tf.graph_util.remove_training_nodes(
        input_graph_def,
        protected_nodes=keep_nodes + [n.name for n in input_graph_def.node if 'nms' in n.name]
    )
    # ops in 'nms' scope must be protected for some reason,
    # but why?

    with tf.gfile.GFile(output_pb_path, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    print('%d ops in the final graph.' % len(output_graph_def.node))


if __name__ == '__main__':
    main()
