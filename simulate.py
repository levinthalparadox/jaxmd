"""
Given the energy function, how do we update the positions of the particles.
Ignoring space to make it cleaner but would use space to get distances and also update positions.
Ignoring initialization.
"""

import jax 
import jax.numpy as jnp
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

def nvt_step_with_rescaling(
    positions: jnp.ndarray,
    old_velocities: jnp.ndarray,
    mass: float,
    energy_fn,
    step_size: float,
    target_temperature: float,
    k_b: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    NVT means we are trying to maintain a constant temperature,
    which is proportional to the average kinetic energy.
    So, naive way is we get difference between target temperature and current temperature
    and scale the kinetic energy and thus velocities. 
    This is problem/unnatural becuase
    """
    new_positions, new_velocities = velocity_verlet_step(
        positions, old_velocities, mass, energy_fn, step_size
    )
    kinetic_energy = 0.5 * mass * jnp.sum(new_velocities ** 2)
    dof = new_velocities.size
    current_temperature = 2.0 * kinetic_energy / (dof * k_b)
    scale = jnp.sqrt(target_temperature / current_temperature)
    new_velocities = new_velocities * scale
    return new_positions, new_velocities


def nose_hoover_single(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    friction_parameter: float,
    mass: float,
    energy_fn,
    time_step: float,
    target_T: float,
    Q: float,
    k_b: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Idea is that instead of arbitrary scaling, we have a thermostat.
    The idea is that the the thermostat heats up andd cool down accordingly
    to make sure the total energy is constant if you include the thermostat.
    It uses a friction parameter to control how fast the thermostat heats up and cools down.
    This can cause problems with the the thermostat becoming too in lock step with the system,
    and so not all the space is properly explored, so sometimes chaining is applied where 
    the thermostata gets its own thermostat.
    """
    dof = velocities.size
    half = 0.5 * time_step
    force = lambda x: -jax.grad(energy_fn)(x)
    current_kinetic_energy = 0.5 * mass * jnp.sum(velocities * velocities)
    temperature_error = 2.0 * current_kinetic_energy - dof * k_b * target_T
    friction_parameter = friction_parameter + half * temperature_error / Q
    velocities = velocities * jnp.exp(-friction_parameter * half)
    velocities = velocities + half * force(positions) / mass
    positions = positions + time_step * velocities
    velocities = velocities + half * force(positions) / mass
    velocities = velocities * jnp.exp(-friction_parameter * half)
    current_kinetic_energy = 0.5 * mass * jnp.sum(velocities * velocities)
    temperature_error = 2.0 * current_kinetic_energy - dof * k_b * target_T
    friction_parameter = friction_parameter + half * temperature_error / Q
    return positions, velocities, friction_parameter

def nvt_langevian_step(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    mass: float,
    energy_fn,
    time_step: float,
    friction_constant: float,
    target_temperature: float,
    key: jax.random.PRNGKey,
    k_b: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
    """
    Instead of using a thermostat, we use the fluctuation dissipation theorem.
    We have some friction but this constant is preset. Additionally, we add some random
    energy to the system to maintain the target temperature.

    Key idea is that the friction is self-regulating. If the system is too hot, the friction will be high
    and if the system is too cold, the friction will be low. However, the friction removes energy
    from the system so we must add some random energy.
    """
    force_fn = lambda x: -jax.grad(energy_fn)(x)
    half_dt = 0.5 * time_step
    exp_gamma = jnp.exp(-friction_constant * half_dt)
    noise_scale = jnp.sqrt((1.0 - exp_gamma**2) * k_b * target_temperature / mass)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    forces = force_fn(positions)
    velocities = velocities + (half_dt / mass) * forces
    velocities = exp_gamma * velocities + noise_scale * jax.random.normal(subkey1, velocities.shape)
    positions = positions + half_dt * velocities
    forces_new = force_fn(positions)
    velocities = velocities + (half_dt / mass) * forces_new
    velocities = exp_gamma * velocities + noise_scale * jax.random.normal(subkey2, velocities.shape)
    positions = positions + half_dt * velocities
    return positions, velocities, key