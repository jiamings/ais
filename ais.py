import tensorflow as tf
import numpy as np
from hmc import *
from parzen import log_mean_exp


def get_schedule(num, rad=4):
    if num == 1:
        return np.array([0.0, 1.0])
    t = np.linspace(-rad, rad, num)
    s = 1.0 / (1.0 + np.exp(-t))
    return (s - np.min(s)) / (np.max(s) - np.min(s))


class Model(object):
    def __init__(self, generator, prior, kernel, sigma, num_samples,
                 stepsize=0.01, n_steps=10,
                 target_acceptance_rate=.65, avg_acceptance_slowness=0.9,
                 stepsize_min=0.0001, stepsize_max=0.5, stepsize_dec=0.98, stepsize_inc=1.02):
        """
        The model performs AIS operations.
        :param generator: The generator supports __call__(z), input_dim, output_dim
        :param prior: The prior is the starting distribution of p(z), either p(z) or q(z|x).
        :param kernel: The kernel defines the "likelihood" of p(x|z), under the parzen approximation.
        :param sigma: Variance parameter for the parzen window
        :param num_samples: Number of samples to sample from in order to estimate the likelihood.

        The following are parameters for HMC.
        :param stepsize:
        :param n_steps:
        :param target_acceptance_rate:
        :param avg_acceptance_slowness:
        :param stepsize_min:
        :param stepsize_max:
        :param stepsize_dec:
        :param stepsize_inc:
        """
        self.sess = tf.Session()
        self.generator = generator
        self.prior = prior
        self.kernel = kernel
        self.x = tf.placeholder(tf.float32, [None, self.generator.output_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.generator.input_dim], name='z')
        self.zv = None
        self.batch_size = tf.shape(self.x)[0]
        self.num_samples = num_samples
        self.sigma = sigma
        self.t = tf.placeholder(tf.float32, [], name='t')
        self.lld = tf.reshape(-self.energy_fn(self.z), [num_samples, self.batch_size])

        self.stepsize = tf.Variable(stepsize)
        self.avg_acceptance_rate = tf.Variable(target_acceptance_rate)

        self.accept, self.final_pos, self.final_vel = hmc_move(
            self.z,
            self.energy_fn,
            stepsize,
            n_steps
        )

        self.new_z, self.updates = hmc_updates(
            self.z,
            self.stepsize,
            avg_acceptance_rate=self.avg_acceptance_rate,
            final_pos=self.final_pos,
            accept=self.accept,
            stepsize_min=stepsize_min,
            stepsize_max=stepsize_max,
            stepsize_dec=stepsize_dec,
            stepsize_inc=stepsize_inc,
            target_acceptance_rate=target_acceptance_rate,
            avg_acceptance_slowness=avg_acceptance_slowness
        )

        self.sess.run(tf.global_variables_initializer())

    def step(self, x, t):
        new_z, accept, vel, _ = self.sess.run([self.new_z, self.accept, self.final_vel, self.updates], feed_dict={self.t: t, self.x: x, self.z: self.zv})
        self.zv = new_z
        return accept

    def log_likelihood(self, x, t):
        return self.sess.run(self.lld, feed_dict={self.t: t, self.x: x, self.z: self.zv})

    def energy_fn(self, z):
        mu = self.generator(z)
        mu = tf.reshape(mu, [self.num_samples, self.batch_size, self.generator.output_dim])
        e = self.prior.logpdf(z) + self.t * tf.reshape(self.kernel.logpdf(self.x, mu, self.sigma), [self.num_samples * self.batch_size])
        return -e

    def ais(self, x, schedule):
        w = 0.0
        self.zv = np.random.normal(0.0, 1.0, [x.shape[0] * self.num_samples, self.generator.input_dim])
        for (t0, t1) in zip(schedule[:-1], schedule[1:]):
            new_u = self.log_likelihood(x, t1)
            prev_u = self.log_likelihood(x, t0)
            w += new_u - prev_u
            print(self.sess.run(self.kernel.logpdf(self.x, tf.reshape(self.generator(self.z), [self.num_samples, self.batch_size, self.generator.output_dim]), self.sigma),
                                feed_dict={self.x: x, self.z: self.zv}))
            accept = self.step(x, t1)
            print(np.mean(accept))
        lld = np.squeeze(log_mean_exp(w, axis=0), axis=0)
        return lld