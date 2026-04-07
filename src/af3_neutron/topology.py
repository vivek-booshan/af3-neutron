import io
import logging
import numpy as np
import jax.numpy as jnp

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import hydride

def build_oracle_from_af3_result(inference_result):
    """
    Takes the final predicted AF3 structure, uses Hydride to build the full 
    Crystallographic Oracle (including dropped hydrogens), and extracts the Z-matrix.
    """
    logging.info("Building Hydride Oracle from AF3 output state...")
    
    # 1. Load AF3's predicted state
    cif_file = pdbx.CIFFile.read(io.StringIO(inference_result.cif_string))
    af3_atoms = pdbx.get_structure(cif_file, model=1)
    
    # Fast lookup for AF3 atom indices
    af3_lookup = {}
    for i in range(af3_atoms.array_length()):
        key = (af3_atoms.chain_id[i], af3_atoms.res_id[i], af3_atoms.atom_name[i])
        af3_lookup[key] = i
        
    # 2. Let Hydride build the Oracle
    af3_atoms.bonds = struc.connect_via_residue_names(af3_atoms)
    if "charge" not in af3_atoms.get_annotation_categories():
        af3_atoms.add_annotation("charge", dtype=int)
        af3_atoms.charge[:] = 0 
        
    oracle_atoms, _ = hydride.add_hydrogen(af3_atoms)
    oracle_atoms.coord = hydride.relax_hydrogen(oracle_atoms)
    num_oracle_atoms = oracle_atoms.array_length()
    
    # 3. Build the Decoupled Mappings
    bonds, _ = oracle_atoms.bonds.get_all_bonds()

    rotor_table = {
        "target_idx": [], "parent_idx": [], 
        "grandparent_idx": [], "greatgrand_idx": [],
        "ideal_r": [], "ideal_theta": []
    }
    
    oracle_heavy_indices = []
    af3_source_indices = []

    for i in range(num_oracle_atoms):
        is_hydrogen = (oracle_atoms.element[i] == "H")
        h_key = (oracle_atoms.chain_id[i], oracle_atoms.res_id[i], oracle_atoms.atom_name[i])
        
        # If it's a heavy atom or a rigid hydrogen that AF3 kept, map directly
        if h_key in af3_lookup:
            oracle_heavy_indices.append(i)
            af3_source_indices.append(af3_lookup[h_key])
            continue
            
        # If it's a dropped hydrogen, build its NeRF matrix
        if not is_hydrogen: continue
            
        p_indices = bonds[i][bonds[i] != -1]
        if len(p_indices) == 0: continue
        p_i = p_indices[0]
        
        gp_indices = bonds[p_i][bonds[p_i] != -1]
        gp_indices = gp_indices[gp_indices != i]
        if len(gp_indices) == 0: continue 
        gp_i = gp_indices[0]
        
        ggp_indices = bonds[gp_i][bonds[gp_i] != -1]
        ggp_indices = ggp_indices[ggp_indices != p_i]
        if len(ggp_indices) == 0: continue
        ggp_i = ggp_indices[0]
        
        p_key = (oracle_atoms.chain_id[p_i], oracle_atoms.res_id[p_i], oracle_atoms.atom_name[p_i])
        gp_key = (oracle_atoms.chain_id[gp_i], oracle_atoms.res_id[gp_i], oracle_atoms.atom_name[gp_i])
        ggp_key = (oracle_atoms.chain_id[ggp_i], oracle_atoms.res_id[ggp_i], oracle_atoms.atom_name[ggp_i])
        
        # Anchors MUST exist in the AF3 network output
        if not (p_key in af3_lookup and gp_key in af3_lookup and ggp_key in af3_lookup):
            continue
            
        c_h, c_p, c_gp = oracle_atoms.coord[i], oracle_atoms.coord[p_i], oracle_atoms.coord[gp_i]
        v_hp, v_gpp = c_h - c_p, c_gp - c_p
        
        r_ideal = np.linalg.norm(v_hp)
        cos_theta = np.dot(v_hp, v_gpp) / (r_ideal * np.linalg.norm(v_gpp) + 1e-8)
        theta_ideal = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        
        rotor_table["target_idx"].append(i) # Target is Oracle index
        rotor_table["parent_idx"].append(af3_lookup[p_key]) # Anchors are AF3 indices
        rotor_table["grandparent_idx"].append(af3_lookup[gp_key])
        rotor_table["greatgrand_idx"].append(af3_lookup[ggp_key])
        rotor_table["ideal_r"].append(r_ideal)
        rotor_table["ideal_theta"].append(theta_ideal)

    logging.info(f"Mapped {len(oracle_heavy_indices)} atoms directly to AF3 state.")
    logging.info(f"Delegated {len(rotor_table['target_idx'])} missing protons to JAX NeRF.")

    rotor_table_jax = {k: jnp.array(v, dtype=jnp.float32 if "ideal" in k else jnp.int32) for k, v in rotor_table.items()}
    mapping_jax = {
        "oracle_heavy": jnp.array(oracle_heavy_indices, dtype=jnp.int32),
        "af3_source": jnp.array(af3_source_indices, dtype=jnp.int32),
        "num_oracle_atoms": num_oracle_atoms
    }
    
    return rotor_table_jax, mapping_jax
