import tensorflow as tf
import numpy as np
from ais.hmc import *

def get_schedule(num, rad=4):
    if num == 1:
        return np.array([0.0, 1.0])
    t = np.linspace(-rad, rad, num)
    s = 1.0 / (1.0 + np.exp(-t))
    return (s - np.min(s)) / (np.max(s) - np.min(s))

def log_mean_exp(x, axis=None):
    m = tf.reduce_max(x, axis=axis, keep_dims=True)
    return m + tf.log(tf.reduce_mean(tf.exp(x - m), axis=axis, keep_dims=True))

class AIS(object):
    def __init__(self, x_ph, log_likelihood_fn, dims, num_samples,
                 stepsize=0.01, n_steps=10,
                 target_acceptance_rate=.65, avg_acceptance_slowness=0.9,
                 stepsize_min=0.0001, stepsize_max=0.5, stepsize_dec=0.98, stepsize_inc=1.02):
        """
        The model implements Hamiltonian AIS.
        Developed by @bilginhalil on top of https://github.com/jiamings/ais/

        Example use case:
        logp(x|z) = |integrate over z|{logp(x|z,theta) + logp(z)}
        p(x|z, theta) -> likelihood function p(z) -> prior
        Prior is assumed to be a normal distribution with mean 0 and identity covariance matrix

        :param x_ph: Placeholder for x
        :param log_likelihood_fn: Outputs the logp(x|z, theta), it should take two parameters: x and z
        :param e.g. {'output_dim': 28*28, 'input_dim': FLAGS.d, 'batch_size': 1} :)
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

        self.dims = dims
        self.log_likelihood_fn = log_likelihood_fn
        self.num_samples = num_samples

        self.z_shape = [dims['batch_size'] * self.num_samples, dims['input_dim']]

        tfd = tf.contrib.distributions
        self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros(self.z_shape),
                        scale_diag=tf.ones(self.z_shape))

        self.batch_size = dims['batch_size']
        self.x = x_ph

        self.stepsize = stepsize
        self.avg_acceptance_rate = target_acceptance_rate
        self.n_steps = n_steps
        self.stepsize_min = stepsize_min
        self.stepsize_max = stepsize_max
        self.stepsize_dec = stepsize_dec
        self.stepsize_inc = stepsize_inc
        self.target_acceptance_rate = target_acceptance_rate
        self.avg_acceptance_slowness = avg_acceptance_slowness

    def log_f_i(self, z, t):

        return tf.reshape(- self.energy_fn(z, t), [self.num_samples, self.batch_size])

    def energy_fn(self, z, t):

        e = self.prior.log_prob(z) + t * \
            tf.reshape(self.log_likelihood_fn(self.x, z),
                       [self.num_samples * self.batch_size])

        return -e

    def ais(self, z, schedule):
        """
            :param z: initial samples drawn from prior, with shape [num_samples*batch_size]
            :param schedule: temperature schedule i.e. `p(z)p(x|z)^t`
        """

        index_summation = (tf.constant(0),
                           tf.zeros([self.num_samples, self.batch_size]),
                           tf.cast(z, tf.float32),
                           self.stepsize,
                           self.avg_acceptance_rate
                           )

        items = tf.unstack(tf.convert_to_tensor([[i, t0, t1] for i, (t0, t1) in enumerate(zip(schedule[:-1], schedule[1:]))]))

        def condition(index, summation, z, stepsize, avg_acceptance_rate):
            return tf.less(index, len(schedule)-1)

        def body(index, w, z, stepsize, avg_acceptance_rate):
            item = tf.gather(items, index)
            t0 = tf.gather(item, 1)
            t1 = tf.gather(item, 2)

            new_u = self.log_f_i(z, t1)
            prev_u = self.log_f_i(z, t0)

            w = tf.add(w, new_u - prev_u)

            def run_energy(z):
                e = self.energy_fn(z, t1)
                with tf.control_dependencies([e]):
                    return e

            # New step:
            accept, final_pos, final_vel = hmc_move(
                z,
                run_energy,
                stepsize,
                self.n_steps
            )

            new_z, new_stepsize, new_acceptance_rate = hmc_updates(
                z,
                stepsize,
                avg_acceptance_rate=avg_acceptance_rate,
                final_pos=final_pos,
                accept=accept,
                stepsize_min=self.stepsize_min,
                stepsize_max=self.stepsize_max,
                stepsize_dec=self.stepsize_dec,
                stepsize_inc=self.stepsize_inc,
                target_acceptance_rate=self.target_acceptance_rate,
                avg_acceptance_slowness=self.avg_acceptance_slowness
            )

            return tf.add(index,1), w, new_z, new_stepsize, new_acceptance_rate

        i, w, _, _, _ = tf.while_loop(condition, body, index_summation, parallel_iterations=1, swap_memory=True)

        return tf.squeeze(log_mean_exp(w, axis=0), axis=0)