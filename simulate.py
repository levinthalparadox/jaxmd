"""
Given the energy function, how do we update the positions of the particles.
Ignoring space to make it cleaner but would use space to get distances and also update positions.
"""

import jax 
def euler_step(positions : jnp.ndarray, energy_fn, step_size : float) -> jnp.ndarray:
    """
    Euler step is not that accurate. Ie see Taylor series expansion of energy function.
    Furthermore, it assumes that the force is constant over the time step.
    Furthermore, it is not energy conserving. Think of a ball falling down. Because you 
    use the old velocity/force at start of step, the position will change and does potential 
    energy will change but will be using old velocity.  Ie lets say you start at rest,
    at timestep 1 velocity will change because of the force but the position will still be the same
    as the initial velocity was 0, so we will have a net change in energy. 
    """
    force_fn = lambda x: -jax.grad(energy_fn)(x)
    forces = force_fn(positions)
    new_positions = positions + step_size * forces
    return new_positions

def runge_kutta_step(positions : jnp.ndarray, energy_fn, step_size : float) -> jnp.ndarray:
    """
    Runge-Kutta step doesn't assume that the force is constant over the time step
    and averages stuff so it works well with the Taylor series. It does well to minimize 
    the local error but it is also not energy conserving. 
    """
    force_fn = lambda x: -jax.grad(energy_fn)(x)
    k1 = force_fn(positions)
    k2 = force_fn(positions + step_size * k1 / 2)
    k3 = force_fn(positions + step_size * k2 / 2)
    k4 = force_fn(positions + step_size * k3)
    new_positions = positions + step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return new_positions