import jax
import jax.numpy as jnp
import numpy as np
from af3_neutron.kinematics import safe_norm, generalized_nerf_layer

def test_safe_norm_normal_vector():
    v = jnp.array([[3.0, 4.0, 0.0]])
    n = safe_norm(v)
    assert np.allclose(n, 5.0, atol=1e-3)

def test_safe_norm_zero_vector():
    # A standard jnp.linalg.norm would yield a NaN gradient here.
    # We test that safe_norm returns a tiny finite value instead of 0.
    v = jnp.array([[0.0, 0.0, 0.0]])
    n = safe_norm(v)
    assert n[0, 0] > 0.0
    assert not jnp.isnan(n[0, 0])

def test_nerf_placement():
    # Create a simple right-angle backbone: 
    # P0 at origin, P1 along X, P2 along Y
    heavy_coords = jnp.array([
        [0.0, 0.0, 0.0],  # Great-grandparent (P0)
        [1.0, 0.0, 0.0],  # Grandparent (P1)
        [1.0, 1.0, 0.0],  # Parent (P2)
    ])
    
    rotor_table = {
        "parent_idx": jnp.array([2]),
        "grandparent_idx": jnp.array([1]),
        "greatgrand_idx": jnp.array([0]),
        "ideal_r": jnp.array([1.0]),
        "ideal_theta": jnp.array([90.0]), 
    }
    
    # 0 degrees torsion should place H along the Z-axis relative to the local frame
    chi_angles = jnp.array([0.0])
    
    h_coords = generalized_nerf_layer(heavy_coords, rotor_table, chi_angles)
    
    assert h_coords.shape == (1, 3)
    assert not jnp.any(jnp.isnan(h_coords))
    
def test_nerf_gradient_flow():
    # Ensure gradients can flow through the NeRF layer to the heavy anchors
    def dummy_nerf_loss(heavy, chi, table):
        h = generalized_nerf_layer(heavy, table, chi)
        return jnp.sum(h**2)
        
    grad_fn = jax.value_and_grad(dummy_nerf_loss, argnums=(0, 1))
    
    # Test with perfectly overlapping atoms (the stress test for safe_norm)
    heavy_coords = jnp.zeros((3, 3))
    rotor_table = {
        "parent_idx": jnp.array([2]),
        "grandparent_idx": jnp.array([1]),
        "greatgrand_idx": jnp.array([0]),
        "ideal_r": jnp.array([1.0]),
        "ideal_theta": jnp.array([109.5]), 
    }
    chi_angles = jnp.array([0.0])
    
    loss, (grad_heavy, grad_chi) = grad_fn(heavy_coords, chi_angles, rotor_table)
    
    assert not jnp.isnan(loss)
    assert not jnp.any(jnp.isnan(grad_heavy))
    assert not jnp.any(jnp.isnan(grad_chi))
