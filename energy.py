import jax.numpy as jnp

"""
Energy functions provide a way to calculate the energy of a system given the positions of the particles.
In actual implemention, would try to compute this in parallel and use neighborlists. 
"""
def soft_sphere_potential(
    distance_between_centers,
    repulsion_energy_scale,
    particle_diameter,
    stiffness_exponent
    ): 
    """
    Pure respulsion. If two molecules overlap, they repel each other; otherwise nothing. 
    Stifness exponent controls how stiff the repulsion if overlap is. 
    Think Pauli Exclusion Principle
    """
    distance = jnp.asarray(distance_between_centers)
    is_overlapping = distance < particle_diameter
    overlap_fraction = 1 - distance / particle_diameter
    potential_energy = repulsion_energy_scale * (overlap_fraction ** stiffness_exponent)
    return jnp.where(is_overlapping, potential_energy, 0.0)

def lennard_jones_potential(
    distance_between_centers,
    min_potential,
    equilibrium_distance
    ):
    """
    Add an attractive force to the soft sphere ie simulate something like vanderwaals forces.
    Example is london dispersion force (electron is moving around sometimes can create dipole, 
    will induce dipole in other atom, and then can have attractive force between them)
    If far apart, attractive force dominates; if close, repulsive force dominates.
    Equilibrium distance is the distance at which the attractive and repulsive forces balance out.

    Graph here is helpful https://physicsatmcl.commons.msu.edu/lennard-jones-potential/
    """
    distance = jnp.asarray(distance_between_centers)
    scaled_distance = equilibrium_distance / distance
    repulsive_term = scaled_distance ** 12
    attractive_term = scaled_distance ** 6
    potential_energy = 4 * min_potential * (repulsive_term - attractive_term)
    return potential_energy