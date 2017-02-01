import tensorflow as tf
import numpy as np

from parzen import tf_log_mean_exp


class ParsenDensityEstimator(object):
    def logpdf(self, x, mu, sigma):
        """
        Calculate the logpdf.
        :param x: [batch_size, output_dim]
        :param mu: Shape [num_samples, batch_size, output_dim]
        :param sigma: variance
        :return:
        """
        d = (tf.expand_dims(x, 0) - mu) / sigma
        e = -0.5 * tf.reduce_sum(tf.mul(d, d), axis=2)
        z = tf.to_float(tf.shape(mu)[2]) * tf.log(np.float32(sigma * np.sqrt(np.pi * 2.0)))
        return e - z