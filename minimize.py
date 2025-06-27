"""
Minimize is a way to get into a low energy state of a system before we start simulating. 
"""

import jax.numpy as jnp
from jax import grad


def gradient_descent(positions, energy_fn, max_steps=1000, step_size=0.01, force_tolerance=1e-6):
    """Basic gradient descent minimizer. Very similar to what we use in machine learning.
    Gradient descent is sometimes problematic is because of oscillations especially when
    one dimension is not scaled properly. 
    """
    force_fn = grad(energy_fn)
    for _ in range(max_steps):
        forces = -force_fn(positions)
        positions = positions + step_size * forces 
        if jnp.linalg.norm(forces) < force_tolerance:
            break
    return positions


def fire_minimize(positions, energy_fn, max_steps=1000, dt_max=0.1, dt_min=0.02, force_tolerance=1e-6):
    """Fire tries to let you learn faster without the oscillations of gradient descent.

    Couple key ideas
    1. Has a concept of momentum to reward you for going in the right direction
    2. If you overstep, velocity get reset and also timestep gets smaller
    """
    force_fn = grad(energy_fn)
    alpha = 0.1
    dt = dt_max
    velocity = jnp.zeros_like(positions)
    for _ in range(max_steps):
        forces = -force_fn(positions)
        force_norm = jnp.linalg.norm(forces)
        if force_norm < force_tolerance:
            break
        velocity = velocity + dt * forces
        positions = positions + dt * velocity
        power = jnp.sum(forces * velocity)
        if power > 0:
            velocity = (1 - alpha) * velocity + alpha * forces * jnp.linalg.norm(velocity) / jnp.linalg.norm(forces)
            dt = min(dt * 1.1, dt_max)
            alpha *= 0.99
        else:
            velocity = jnp.zeros_like(velocity)
            dt = max(dt * 0.5, dt_min)
            alpha = 0.1
    return positions