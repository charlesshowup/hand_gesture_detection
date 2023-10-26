import tensorflow._api.v2.compat.v1 as tf
import tf_slim as slim
from src.constants import BATCH_NORM_MOMENTUM
from config import params
from backbone import inference
#from src.net448 import *
#from src.face_net import *
#from src.face_net_64ds import *
#from src.net3 import *
#from src.lffdv1 import *
#from src.net import *
#from src.vovnet_face import *
#from src.vovnet_face_1 import *
'''
if params['quantization_params']['use_quantization_model']:
    from src.model.net_inference_W8bitA4bit import *
else:
    from src.model.net_inference_float import *
'''

class FeatureExtractor:
    def __init__(self, is_training):
        self.is_training = is_training
        if params['model_params']['is_fine_tune_landmark']:
            self.is_training = False
            self.trainable = False
        else:
            self.trainable = True


    def extract_feat(self, images):
        """
        Arguments:
            images: a float tensor with shape [batch_size, height, width, 3],
                a batch of RGB images with pixel values in the range [0, 1].
        Returns:
            a list of float tensors where the ith tensor
            has shape [batch, height_i, width_i, channels_i].
        """

        with tf.name_scope('standardize_input'):
            x = preprocess(images)

        features = inference(x, self.is_training, self.trainable)
        return features

    def get_total_stride(self):
        total_stride = 32
        return total_stride


def preprocess(images):
    """Transform images before feeding them to the network."""
    return (2.0*images) - 1.0

