import jax.numpy as jnp

def safe_norm(x, axis=-1, keepdims=True):
    """
    Calculates the L2 norm safely. 
    Adding epsilon INSIDE the square root prevents NaN gradients when x == 0.
    """
    return jnp.sqrt(jnp.sum(x**2, axis=axis, keepdims=keepdims) + 1e-8)

def generalized_nerf_layer(heavy_coords, rotor_table, chi_angles):
    p2 = heavy_coords[rotor_table["parent_idx"]]       
    p1 = heavy_coords[rotor_table["grandparent_idx"]]  
    p0 = heavy_coords[rotor_table["greatgrand_idx"]]   
    
    v1 = p2 - p1
    v2 = p1 - p0
    
    # Use safe_norm to ensure bulletproof JAX gradients
    z_axis = v1 / safe_norm(v1)
    x_axis_raw = jnp.cross(v2, z_axis)
    x_axis = x_axis_raw / safe_norm(x_axis_raw)
    y_axis = jnp.cross(z_axis, x_axis)
    
    R = jnp.stack([x_axis, y_axis, z_axis], axis=-1)
    
    r = rotor_table["ideal_r"]
    theta = jnp.radians(rotor_table["ideal_theta"])
    
    local_coords = jnp.stack([
        r * jnp.sin(theta) * jnp.cos(chi_angles),
        r * jnp.sin(theta) * jnp.sin(chi_angles),
        -r * jnp.cos(theta)
    ], axis=-1)
    
    global_coords = jnp.einsum('nij,nj->ni', R, local_coords) + p2
    return global_coords
