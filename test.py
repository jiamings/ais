import tensorflow as tf
import numpy as np
import ais
import matplotlib.pyplot as plt
from scipy.stats import norm


class Generator(object):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, z):
        return z * 2 + 3

def kde_logpdf(x, mu, sigma):
    """
    Calculate the kde logpdf.
    :param x: Shape [num_samples*batch_size, output_dim]
    :param mu: Shape [num_samples*batch_size, output_dim]
    :param sigma: variance
    :return: [num_samples*batch_size]
    """
    # K(u) = 1/sqrt(2*pi) * exp(-0.5 * u^2)
    # p(x) = 1/nh * |sum over n, i|{K((x-xi)/h)}
    # logp(x) = -log(nh) + |sum over n, i|{-log(sqrt(2*pi) + 0.5*((mu-xi)/h)^2}
    # logp(x)[i] = -log(h) -n*log(sqrt(2*pi)) -log(sqrt(2*pi) + 0.5*((mu-xi)/h)^2
    # instead of summing it then taking the average of it, we simply keep things separately in a
    # matrix. log_mean_exp will run in the end of AIS method. and we will obtain a shape
    # [batch_size] instead of [batch_size*num_samples]

    d = (x - mu) / sigma
    e = -0.5 * tf.multiply(d, d)

    z = tf.cast(tf.to_float(tf.shape(mu)[1]) *
                tf.log(np.float32(sigma * np.sqrt(np.pi * 2.0))), tf.float32)

    return e - z

generator = Generator(1, 1)

p = norm()

batch_size = 40
num_samples = 10000
x = np.linspace(norm.ppf(0.01, loc=3, scale=2), norm.ppf(0.99, loc=3, scale=2), batch_size)
p1 = norm.pdf(x, loc=3, scale=2)

x_ph = tf.placeholder(tf.float32, [None, 1], name='x')
model = ais.AIS(x_ph, lambda x, z: kde_logpdf(x, generator(z), 1.5),
                  {'input_dim': 1, 'output_dim': 1, 'batch_size': batch_size},
                  num_samples)

xx = np.reshape(x, [batch_size, 1])

schedule = ais.get_schedule(5, rad=4)

#print(schedule)
target = tf.tile(tf.expand_dims(model.x, 0), [model.num_samples, 1,1])
model.x = tf.reshape(target, [model.num_samples * model.batch_size, model.dims['output_dim']])

lld = model.ais(tf.distributions.Normal(loc=[0.], scale=[1.]).sample(num_samples*batch_size),
                schedule)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

p2 = sess.run([lld], {x_ph: xx})
plt.plot(x, p1)
plt.plot(x, np.exp(p2[0]))
plt.show()

