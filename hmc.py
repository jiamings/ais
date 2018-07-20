import tensorflow as tf


def kinetic_energy(v):
    return 0.5 * tf.reduce_sum(tf.multiply(v, v), axis=1)


def hamiltonian(p, v, f):
    return f(p) + kinetic_energy(v)


def metropolis_hastings_accept(energy_prev, energy_next):
    ediff = energy_prev - energy_next
    return (tf.exp(ediff) - tf.random_uniform(tf.shape(energy_prev))) >= 0.0


def simulate_dynamics(initial_pos, initial_vel, stepsize, n_steps, energy_fn):
    def leapfrog(pos, vel, step, i):
        dE_dpos = tf.cast(tf.gradients(tf.reduce_sum(energy_fn(pos)), pos)[0], tf.float32)
        new_vel = vel - step * dE_dpos
        new_pos = pos + step * new_vel

        return [new_pos, new_vel, step, tf.add(i, 1)]

    def condition(pos, vel, step, i):
        return tf.less(i, n_steps)

    dE_dpos = tf.cast(tf.gradients(tf.reduce_sum(energy_fn(initial_pos)), initial_pos)[0], tf.float32)
    vel_half_step = initial_vel - 0.5 * stepsize * dE_dpos
    pos_full_step = initial_pos + stepsize * vel_half_step

    i = tf.constant(0)
    final_pos, new_vel, _, _ = tf.while_loop(condition, leapfrog, [pos_full_step, vel_half_step, stepsize, i])
    dE_dpos = tf.gradients(tf.reduce_sum(energy_fn(final_pos)), final_pos)[0]
    final_vel = new_vel - 0.5 * stepsize * dE_dpos
    return final_pos, final_vel


def hmc_move(initial_pos, energy_fn, stepsize, n_steps):
    initial_vel = tf.random_normal(tf.shape(initial_pos))
    final_pos, final_vel = simulate_dynamics(
        initial_pos=initial_pos,
        initial_vel=initial_vel,
        stepsize=stepsize,
        n_steps=n_steps,
        energy_fn=energy_fn
    )
    accept = metropolis_hastings_accept(
        energy_prev=hamiltonian(initial_pos, initial_vel, energy_fn),
        energy_next=hamiltonian(final_pos, final_vel, energy_fn)
    )
    return accept, final_pos, final_vel


def hmc_updates(initial_pos, stepsize, avg_acceptance_rate, final_pos, accept,
                target_acceptance_rate, stepsize_inc, stepsize_dec,
                stepsize_min, stepsize_max, avg_acceptance_slowness):
    new_pos = tf.where(accept, final_pos, initial_pos)
    new_stepsize_ = tf.where(avg_acceptance_rate > target_acceptance_rate, stepsize_inc, stepsize_dec) * stepsize
    new_stepsize = tf.maximum(tf.minimum(new_stepsize_, stepsize_max), stepsize_min)
    new_acceptance_rate = tf.add(avg_acceptance_slowness * avg_acceptance_rate,
                                 (1.0 - avg_acceptance_slowness) * tf.reduce_mean(tf.to_float(accept)))
    return new_pos, new_stepsize, new_acceptance_rate
