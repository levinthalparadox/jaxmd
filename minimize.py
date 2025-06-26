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