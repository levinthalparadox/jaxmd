import jax.numpy as jnp
from jax import vmap

"""
Energy functions provide a way to calculate the energy of a system given the positions of the particles.
In actual implemention, would try to compute this in parallel and use neighborlists. 
"""
def soft_sphere_potential(
    distance_between_centers : jnp.ndarray,
    repulsion_energy_scale : float,
    particle_diameter : float,
    stiffness_exponent : float
    ): 
    """
    Pure repulsion. If two molecules overlap, they repel each other; otherwise nothing.  
    Stiffness exponent controls how stiff the repulsion if overlap is. 
    Think Pauli Exclusion Principle
    """
    is_overlapping = distance_between_centers < particle_diameter
    overlap_fraction = 1 - distance_between_centers / particle_diameter
    potential_energy = repulsion_energy_scale * (overlap_fraction ** stiffness_exponent)
    return jnp.where(is_overlapping, potential_energy, 0.0)

def lennard_jones_potential(
    distance_between_centers : jnp.ndarray,
    min_potential : float,
    equilibrium_distance : float
    ):
    """
    Add an attractive force to the soft sphere ie simulate something like vanderwaals forces.
    Example is london dispersion force (electron is moving around sometimes can create dipole, 
    will induce dipole in other atom, and then can have attractive force between them)
    If far apart, attractive force dominates; if close, repulsive force dominates.
    Equilibrium distance is the distance at which the attractive and repulsive forces balance out.
    Min potential is the minimum potential energy. Idea is you get closer, attractive forces increase
    but then at some point you get too close and repulsive forces come in. Can think as dissociation energy.
    Graph here is helpful https://physicsatmcl.commons.msu.edu/lennard-jones-potential/
    """
    scaled_distance = equilibrium_distance / distance_between_centers
    repulsive_term = scaled_distance ** 12
    attractive_term = scaled_distance ** 6
    potential_energy = 4 * min_potential * (repulsive_term - attractive_term)
    return potential_energy

def morse_potential(
    distance_between_centers : jnp.ndarray,
    min_potential : float,
    equilibrium_distance : float,
    potential_width_parameter : float
    ):
    """
    Better for bonding than lennard jones. More realistic repulsion than 12th power.
    Potential width parameter controls how quickly energy increases as distance changes from minimum potential_distance
    """
    exponent = -potential_width_parameter * (distance_between_centers - equilibrium_distance)
    potential_energy = min_potential * (1 - jnp.exp(exponent))**2
    return potential_energy


def embedded_atom_model_potential(
    positions: jnp.ndarray,
    pair_potential_func,
    electron_density_func,
    embedding_func
    ):
    """
    EAM is different because can't just treat each pair independently, ie think metals sharing electrons
    Pair potential functional is generally used for repulsive forces, pauli exclusion principle (think soft sphere.
    What we do is we calculate electron density at each atom and then use that to calculate embedding energy.
    This embedding energy is non-linear, so this is main reason we can't do pairwise functions
    """
    dist_matrix = jnp.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    mask = jnp.eye(positions.shape[0], dtype=bool)
    pair_potentials = jnp.where(mask, 0.0, pair_potential_func(dist_matrix))
    total_pair_energy = jnp.sum(pair_potentials) / 2.0
    electron_densities = jnp.where(mask, 0.0, electron_density_func(dist_matrix))
    host_electron_density_per_atom = jnp.sum(electron_densities, axis=1)
    embedding_energy_per_atom = vmap(embedding_func)(host_electron_density_per_atom)
    total_embedding_energy = jnp.sum(embedding_energy_per_atom)
    total_energy = total_pair_energy + total_embedding_energy
    return total_energy