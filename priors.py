import tensorflow as tf
import numpy as np

from parzen import tf_log_mean_exp


class NormalPrior(object):
    def logpdf(self, x):
        d = tf.expand_dims(x, 1)
        e = tf_log_mean_exp(-0.5 * tf.reduce_sum(tf.mul(d, d), axis=2), axis=1)
        e = tf.squeeze(e, axis=1)
        z = tf.to_float(tf.shape(x)[1]) * tf.log(np.float32(1.0 * np.sqrt(np.pi * 2.0)))
        return e - z
