"""
Given the energy function, how do we update the positions of the particles.
Ignoring space to make it cleaner but would use space to get distances and also update positions.
"""

import jax 
from typing import Tuple
def euler_step(positions : jnp.ndarray, old_velocities : jnp.ndarray, mass : float, energy_fn, step_size : float) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    new_velocities = old_velocities + step_size * forces / mass
    new_positions = positions + step_size * new_velocities
    return new_positions, new_velocities

def runge_kutta_step(positions : jnp.ndarray, old_velocities : jnp.ndarray, mass : float, energy_fn, step_size : float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Runge-Kutta step doesn't assume that the force is constant over the time step
    and averages stuff so it works well with the Taylor series. It does well to minimize 
    the local error but it is also not energy conserving. 
    """
    force_fn = lambda x: -jax.grad(energy_fn)(x)
    k1_forces = force_fn(positions)
    k1_velocities = old_velocities + step_size * k1_forces / mass
    k2_forces = force_fn(positions + step_size * k1_velocities / 2)
    k2_velocities = old_velocities + step_size * k2_forces / mass
    k3_forces = force_fn(positions + step_size * k2_velocities / 2)
    k3_velocities = old_velocities + step_size * k3_forces / mass
    k4_forces = force_fn(positions + step_size * k3_velocities)
    k4_velocities = old_velocities + step_size * k4_forces / mass
    new_positions = positions + step_size * (k1_velocities + 2 * k2_velocities + 2 * k3_velocities + k4_velocities) / 6
    new_velocities = old_velocities + step_size * (k1_forces + 2 * k2_forces + 2 * k3_forces + k4_forces) / 6
    return new_positions, new_velocities

def velocity_verlet_step(positions : jnp.ndarray, old_velocities : jnp.ndarray, mass : float, energy_fn, step_size : float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Velocity Verlet step is conserving, Intutition, is position is updated with half velocity 
    so same as in forward vs backward so is generally energy conserving. 
    """
    force_fn = lambda x: -jax.grad(energy_fn)(x)
    forces = force_fn(positions)    
    velocities_half = old_velocities + 0.5 * (forces / mass) * step_size
    new_positions = positions + velocities_half * step_size
    new_forces = force_fn(new_positions)
    new_velocities = velocities_half + 0.5 * (new_forces / mass) * step_size
    return new_positions, new_velocities

def nve_step(positions : jnp.ndarray, old_velocities : jnp.ndarray, mass : float, energy_fn, step_size : float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    NVE means number of particles is constant, energy is constant and volume is constant. Space and
    number of particles in our non-free space and see above for velocity verlet being energy conserving.
    """
    new_positions, new_velocities = velocity_verlet_step(positions, old_velocities, mass, energy_fn, step_size)
    return new_positions, new_velocities
    