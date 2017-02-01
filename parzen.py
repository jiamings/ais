import tensorflow as tf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def log_mean_exp(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True)
    return m + np.log(np.mean(np.exp(x - m), axis=axis, keepdims=True))


def tf_log_mean_exp(x, axis=None):
    m = tf.reduce_max(x, axis=axis, keep_dims=True)
    return m + tf.log(tf.reduce_mean(tf.exp(x - m), axis=axis, keep_dims=True))


def tf_parzen(x, mu, sigma):
    d = (tf.expand_dims(x, 1) - tf.expand_dims(mu, 0)) / sigma
    e = tf_log_mean_exp(-0.5 * tf.reduce_sum(tf.mul(d, d), axis=2), axis=1)
    e = tf.squeeze(e, axis=1)
    z = tf.to_float(tf.shape(mu)[1]) * tf.log(sigma * np.sqrt(np.pi * 2.0))
    return e - z


class ParsenDensityEstimator(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32)
        self.mu = tf.placeholder(tf.float32)
        self.sigma = tf.placeholder(tf.float32, [])
        self.ll = tf_parzen(self.x, self.mu, self.sigma)
        self.sess = tf.Session()

    def logpdf(self, x, mu, sigma, sess=None):
        sess = sess or self.sess
        return sess.run(self.ll, feed_dict={self.x: x, self.mu: mu, self.sigma: sigma})


if __name__ == '__main__':
    p = norm()
    x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
    p1 = norm.pdf(x)

    xx = np.reshape(x, [100, 1])
    e = ParsenDensityEstimator()
    m = np.random.normal(0.0, 1.0, [100000, 1])
    s = 0.25
    p2 = e.logpdf(xx, m, s)
    p2 = np.reshape(p2, [100])
    p2 = np.exp(p2)

    plt.plot(x, p1)
    plt.plot(x, p2)
    plt.show()
