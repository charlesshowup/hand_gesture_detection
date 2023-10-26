import tensorflow._api.v2.compat.v1 as tf
import tf_slim as slim

def add_weight_decay(weight_decay):
    """Add L2 regularization to all (or some) trainable kernel weights."""
    weight_decay = tf.constant(
        weight_decay, tf.float32,
        [], 'weight_decay'
    )
    trainable_vars = tf.trainable_variables()
    kernels = [v for v in trainable_vars if 'weights' in v.name]
    for K in kernels:
        x = tf.multiply(weight_decay, tf.nn.l2_loss(K))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, x)


def get_multi_loss_org(loss_list):
    loss_sum = 0
    for i in range(len(loss_list)):
        log_var = slim.variable('log_var_%d' % i,
                                dtype=tf.float32,
                                shape=[],
                                initializer=tf.initializers.random_uniform(minval=0.0, maxval=1.0))
        #log_var = tf.clip_by_value(log_var, 0.0, 100.0)
        tf.summary.scalar('log_var_%d' % i, log_var)
        precision = tf.exp(-log_var)
        loss = loss_list[i]
        loss_sum += tf.reduce_sum(precision * loss + log_var)
    return loss_sum


def get_multi_loss(loss_dict):
    loss_sum = 0
    for loss_name, loss in loss_dict.items():
        var = slim.variable('var_%s' % loss_name,
                                dtype=tf.float32,
                                shape=[],
                                #initializer=tf.initializers.random_uniform(minval=0.2, maxval=1.0))
                                initializer=tf.initializers.ones())

        var = tf.clip_by_value(var, 0.0, 100.0)
        tf.summary.scalar('var_%s' % loss_name, var)
        precision = tf.divide(1.0, var)
        loss_sum += tf.reduce_sum(precision * loss + tf.log(1.0+var))
    return loss_sum
