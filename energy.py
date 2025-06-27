import jax.numpy as jnp

def soft_sphere_potential(
    distance_between_centers,
    repulsion_energy_scale,
    particle_diameter,
    stiffness_exponent
): """
   Pure respulsion. If two molecules overlap, they repel each other.
   Stifness exponent controls how server 
   """
    distance = jnp.asarray(distance_between_centers)
    is_overlapping = distance < particle_diameter
    overlap_fraction = 1 - distance / particle_diameter
    potential_energy = repulsion_energy_scale * (overlap_fraction ** stiffness_exponent)
    return jnp.where(is_overlapping, potential_energy, 0.0)