"""
Spaces provide layer of abstraction of space to compute distances between particles.

Key functions:
- displacement_fn: computes displacement between two particles
- shift_fn: shifts a particle by a given displacement
"""

import jax.numpy as jnp

# Create abstract base class for space defining two methods

class Space:
    def displacement_fn(self, point_a: jnp.ndarray, point_b: jnp.ndarray) -> jnp.ndarray:
        pass

    def shift_fn(self, point: jnp.ndarray, displacement: jnp.ndarray) -> jnp.ndarray:
        pass

class FreeSpace(Space):
    """
    Free space is a space where there are no boundaries.  
    Generally, this is not great because particles can fly away
    and be far from other particles. Additionally, particles on boundary 
    don't have neighbors on both sides but internal particles do. 
    """
    def displacement_fn(self, point_a: jnp.ndarray, point_b: jnp.ndarray) -> jnp.ndarray:
        return point_b - point_a

    def shift_fn(self, point: jnp.ndarray, displacement: jnp.ndarray) -> jnp.ndarray:
        return point + displacement

class PeriodicSpace(Space):
    """
    Periodic space is when each side of the box is connected to the opposite side.
    Can think of it like pacman game. The reason why we don't have a boundary ie 
    that we reflect off because then we had surface tension unexpectedly here. 

    We have 4 cases:
    1.  If the distance is positive and less than half the box size, then we can just return the distance
    2.  If the distance is positive and greater than half the box size, then we need to wrap around the box
    3.  If the distance is negative and greater than half the box size, then we need to wrap around the box
    4.  If the distance is negative and less than half the box size, then we can just return the distance

    We can use the round division to get the wrapped distance. 
    If distance is less than half the box size the round will be 0 and we can just return the distance.
    If distance is greater than half the box size, then the positive distance will wrap around and become negative
    and if the distance was initially negative, then it will wrap around and become positive. 
    """
    def __init__(self, box_size: jnp.ndarray):
        self.box_size = box_size

    def displacement_fn(self, point_a: jnp.ndarray, point_b: jnp.ndarray) -> jnp.ndarray:
        initial_distance = point_b - point_a
        wrapped_distance = initial_distance - self.box_size * jnp.round(initial_distance / self.box_size)
        return wrapped_distance   # This is the actual distance between the two points

    def shift_fn(self, point: jnp.ndarray, displacement: jnp.ndarray) -> jnp.ndarray:
        """
        Allows wraparound by modding the displacement by the box size.
        """
        shifted = point + displacement
        shifted = shifted % self.box_size
        return shifted

class PeriodicSpaceGeneral(Space):
    """
    This is for non-cubic spaces. The basis vectors are no longer orthogonal 
    and therefore our simple wraparound modulo tricks won't work as 
    moving along one direction will affect the other directions. 
    To solve this we convert them to separate orthogonal vectors and 
    wrap around each of them separately and then convert them to 
    cartesian coordinates. This is super useful for non-cubic crystals.
    """
    def __init__(self,  box_matrix: jnp.ndarray):
        self.box_matrix = box_matrix

    
    def _convert_to_cartesian_coordinates(self, point_in_fractional: jnp.ndarray) -> jnp.ndarray:
        """
        Each column in box matrix is a basis vector represented in cartesian coordinates.
        So, we multiply the matrix by the point_in_fractional to get the cartesian coordinates.
        """
        return jnp.dot(self.box_matrix, point_in_fractional)

    def _convert_to_fractional_coordinates(self, point_in_cartesian: jnp.ndarray) -> jnp.ndarray:
        """
        Converts a point in cartesian coordinates to fractional coordinates. Basically,
        the inverse step of the cartesian to fractional conversion.
        """
        return jnp.dot(jnp.linalg.inv(self.box_matrix), point_in_cartesian)


    def displacement_fn(self, point_a_cartesian: jnp.ndarray, point_b_cartesian: jnp.ndarray) -> jnp.ndarray:
        initial_cartesian_distance = point_b_cartesian - point_a_cartesian
        initial_fractional_distance = self._convert_to_fractional_coordinates(initial_cartesian_distance)
        # Fractional distance can think of it like unit cell so don't have to worry about its size
        wrapped_distance = initial_fractional_distance - jnp.round(initial_fractional_distance)
        wrapped_cartesian_distance = self._convert_to_cartesian_coordinates(wrapped_distance)
        return wrapped_cartesian_distance
    
    def shift_fn(self, point_cartesian: jnp.ndarray, displacement_cartesian: jnp.ndarray) -> jnp.ndarray:
        shifted_cartesian = point_cartesian + displacement_cartesian
        shifted_fractional = self._convert_to_fractional_coordinates(shifted_cartesian)
        wrapped_fractional = shifted_fractional % 1.0 
        return self._convert_to_cartesian_coordinates(wrapped_fractional)