# Mini JAX-MD

A minimal molecular dynamics simulation library built with JAX, focusing on particle interactions and energy minimization.

## Overview

This project implements core molecular dynamics concepts including:

- **Spatial representations** (`spaces.py`) - Free space, periodic boundaries, and general periodic spaces for non-cubic systems
- **Neighbor finding** (`partition.py`) - Pairwise and cell-list methods for efficient neighbor detection
- **Energy minimization** (`minimize.py`) - Gradient descent and FIRE algorithms for system relaxation

## Features

### Space Types
- `FreeSpace`: Unbounded simulation space
- `PeriodicSpace`: Cubic periodic boundary conditions
- `PeriodicSpaceGeneral`: Non-cubic periodic boundaries with arbitrary basis vectors

### Neighbor Detection
- Naive O(N²) pairwise distance calculation
- Cell-list method with skin thickness for efficiency
- Configurable cutoff distances

### Minimization Algorithms
- Gradient descent with configurable step size and convergence criteria
- FIRE (Fast Inertial Relaxation Engine) for faster convergence without oscillations

## Requirements

- JAX
- NumPy

## Usage

```python
import jax.numpy as jnp
from spaces import PeriodicSpace
from minimize import fire_minimize

# Create periodic space
box_size = jnp.array([10.0, 10.0, 10.0])
space = PeriodicSpace(box_size)

# Initialize particle positions
positions = jnp.random.uniform(0, 10, (100, 3))

# Define energy function and minimize
def energy_fn(positions):
    # Your energy calculation here
    pass

minimized_positions = fire_minimize(positions, energy_fn)
```

## Notes

Based on research notes from GNOME (Graph Networks for Materials Exploration) paper, exploring neural network approaches for materials discovery and structure prediction.