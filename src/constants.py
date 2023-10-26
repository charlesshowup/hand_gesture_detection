import tensorflow._api.v2.compat.v1 as tf

# a small value
#EPSILON = 1e-8
EPSILON = 1e-14

# this is used when we are doing box encoding/decoding
SCALE_FACTORS = [10.0, 10.0, 5.0, 5.0] # times SCALE_FACTORS equals divide by 1/SCALE_FACTORS
SCALE_FACTOR_LANDMARK = 15.0
# you can read about them here:
# github.com/rykov8/ssd_keras/issues/53
# github.com/weiliu89/caffe/issues/155

# here are input pipeline settings.
# you need to tweak these numbers for your system,
# it can accelerate training
SHUFFLE_BUFFER_SIZE = 15000
NUM_THREADS = 8
# read here about the buffer sizes:
# stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle

# images are resized before feeding them to the network
RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR

# threshold for IoU when creating training targets
#MATCHING_THRESHOLD = 0.35
MATCHING_THRESHOLD = 0.30

# this is used in tf.map_fn when creating training targets or doing NMS
PARALLEL_ITERATIONS = 8

# this can be important
BATCH_NORM_MOMENTUM = 0.9

