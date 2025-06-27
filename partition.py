
import jax.numpy as jnp

"""
We can make the assumption to ignore stuff that is further than a certain cutoff distance.
"""



def get_neighbors_pairwise(positions : jnp.ndarray, cutoff : float) -> jnp.ndarray:
    """
    This naively computes the distance between all pairs of particles.
    It also ignores self-interaction.
    """
    diff = positions[:, None, :] - positions[None, :, :]
    distances = jnp.sqrt(jnp.sum(diff**2, axis=-1))
    mask = (distances < cutoff) & (distances > 0)
    neighbors = jnp.where(mask)
    return neighbors


def find_neighboring_cells(cell_index : jnp.ndarray, num_cells_in_dimension : int) -> jnp.ndarray:
    """Find all neighboring cells for a given cell index.
    We are assuming a wrap around to support periodic shapes but 
    we can remove the wrap around and just clip if we want to support 
    non-periodic shapes.
    """
    offsets = jnp.array([
        [x_offset, y_offset, z_offset]
        for x_offset in [-1, 0, 1]
        for y_offset in [-1, 0, 1]
        for z_offset in [-1, 0, 1]
    ])
    # Ensure wrap around by using modulo operation
    neighbors_index = (cell_index + offsets) % num_cells_in_dimension
    return neighbors_index

@dataclass 
class PreviousCellList:
    cell_list : jnp.ndarray
    previous_positions : jnp.ndarray | None
    existing_neighbors : jnp.ndarray | None

"""
WORK IN PROGRESS
Basic logic is you segmented the space into a grid with cell with cutoff_length.
You know that stuff within cutoff has to be one of our neighboring cells so just compare 
with those. Additionally, to speed up more, check how much particles have moved since last time.
Make the cells a little bigger than cutoff and if the particles have moved less than the skin thickness, 
since the particles have moved less than the half of the skin thickness, we can just use the previous neighbors.
"""
    
def get_neighbors_cell_list(positions : jnp.ndarray, cutoff : float, box_size : jnp.ndarray, skin_thickness : float, previous_cell_list : PreviousCellList | None = None) -> PreviousCellList:

    cell_size = cutoff + skin_thickness
    num_cells_in_dimension = jnp.ceil(box_size / cell_size).astype(jnp.int32)
    cell_indices = jnp.floor(positions / cell_size).astype(jnp.int32)
    if previous_cell_list is not None:
        diff_in_positions = positions - previous_cell_list.previous_positions
        positions_distance = jnp.sqrt(jnp.sum(diff_in_positions**2))
        max_distance = jnp.max(positions_distance)
        if max_distance < skin_thickness * 0.5:
            return previous_cell_list            
        
    cell_list = build_cell_list(cell_indices, num_cells_in_dimension)
    for cell_index in cell_list:
        neighbors_index = find_neighboring_cells(cell_index, num_cells_in_dimension)
        points_in_cell = cell_indices[neighbors_index]
        for point_index in points_in_cell:
    