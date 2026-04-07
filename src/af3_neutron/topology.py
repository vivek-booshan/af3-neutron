import io
import logging
import dataclasses
import numpy as np
import jax.numpy as jnp

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import hydride

from alphafold3.constants import chemical_components
from alphafold3.model.atom_layout import atom_layout

def inject_superset_topology_and_get_rotors(cleaned_struc, ccd, fold_input):
    """
    Hacks AF3's layout to keep all protons, extracts the template from the 
    fold_input, and uses Hydride to build the JAX kinematic Z-matrix.
    """
    logging.info("Injecting Super-Set Topology...")
    residues = atom_layout.residues_from_structure(cleaned_struc)
    
    # 1. Override the deprotonation rules with empty sets to keep ALL hydrogens
    num_res = residues.shape[0]
    super_set_deprotonation = np.array([set() for _ in range(num_res)], dtype=object)
    residues_super_set = dataclasses.replace(residues, deprotonation=super_set_deprotonation)
    
    # 2. Generate the flat layout WITH hydrogens
    flat_layout = atom_layout.make_flat_atom_layout(
        residues_super_set, ccd, with_hydrogens=True, skip_unk_residues=True
    )
    
    # 3. Build the JAX Rotor Table by extracting templates from fold_input
    logging.info("Extracting Templates and Building Kinematic Rotor Table...")
    rotor_table = _build_rotor_table_from_templates(fold_input, flat_layout)
    
    return flat_layout, rotor_table

def _build_rotor_table_from_templates(fold_input, flat_layout):
    """
    Parses the mmCIF string from the fold_input template, builds physical 
    hydrogens using Hydride, and maps the internal NeRF coordinates (r, theta)
    back to the AlphaFold 3 flat layout tensor.
    """
    # 1. Find the first valid template mmCIF in the input
    template_cif_string = None
    for chain in fold_input.chains:
        if hasattr(chain, 'templates') and chain.templates:
            for t in chain.templates:
                if t.mmcif:
                    template_cif_string = t.mmcif
                    break
        if template_cif_string:
            break

    if not template_cif_string:
        logging.warning("No template mmCIF found in fold_input. Returning empty rotor table.")
        return _empty_rotor_table()

    # 2. Load template into Biotite
    cif_file = pdbx.CIFFile.read(io.StringIO(template_cif_string))
    atoms = pdbx.get_structure(cif_file, model=1)
    
    # Strip any messy input Hs so Hydride can build them perfectly from scratch
    atoms = atoms[atoms.element != "H"] 
    
    # 3. Hydride Preparation (Bonds and Charges)
    atoms.bonds = struc.connect_via_residue_names(atoms)
    if "charge" not in atoms.get_annotation_categories():
        atoms.add_annotation("charge", dtype=int)
        atoms.charge[:] = 0 
        
    # 4. Add and Relax Hydrogens
    logging.info("Hydride: Adding and relaxing missing hydrogens...")
    atoms, _ = hydride.add_hydrogen(atoms)
    atoms.coord = hydride.relax_hydrogen(atoms)
    
    # 5. Map AF3 flat_layout to Biotite indices
    af3_lookup = {}
    for i in range(flat_layout.shape[0]):
        key = (flat_layout.chain_id[i], flat_layout.res_id[i], flat_layout.atom_name[i])
        af3_lookup[key] = i
        
    # 6. Extract the Z-Matrix
    rotor_table = {
        "target_idx": [], "parent_idx": [], 
        "grandparent_idx": [], "greatgrand_idx": [],
        "ideal_r": [], "ideal_theta": []
    }
    
    bonds, _ = atoms.bonds.get_all_bonds()
    
    for i in range(atoms.array_length()):
        if atoms.element[i] != "H":
            continue
            
        h_key = (atoms.chain_id[i], atoms.res_id[i], atoms.atom_name[i])
        if h_key not in af3_lookup: continue
        h_idx = af3_lookup[h_key]
        
        # Traverse Biotite Graph: Parent
        p_indices = bonds[i][bonds[i] != -1]
        if len(p_indices) == 0: continue
        p_i = p_indices[0]
        
        # Traverse Biotite Graph: Grandparent
        gp_indices = bonds[p_i][bonds[p_i] != -1]
        gp_indices = gp_indices[gp_indices != i]
        if len(gp_indices) == 0: continue # E.g., Water
        gp_i = gp_indices[0]
        
        # Traverse Biotite Graph: Great-Grandparent
        ggp_indices = bonds[gp_i][bonds[gp_i] != -1]
        ggp_indices = ggp_indices[ggp_indices != p_i]
        if len(ggp_indices) == 0: continue
        ggp_i = ggp_indices[0]
        
        # Map Anchors to AF3 Layout
        p_key = (atoms.chain_id[p_i], atoms.res_id[p_i], atoms.atom_name[p_i])
        gp_key = (atoms.chain_id[gp_i], atoms.res_id[gp_i], atoms.atom_name[gp_i])
        ggp_key = (atoms.chain_id[ggp_i], atoms.res_id[ggp_i], atoms.atom_name[ggp_i])
        
        if not (p_key in af3_lookup and gp_key in af3_lookup and ggp_key in af3_lookup):
            continue
            
        # 7. Measure the empirical r and theta
        c_h = atoms.coord[i]
        c_p = atoms.coord[p_i]
        c_gp = atoms.coord[gp_i]
        
        v_hp = c_h - c_p
        v_gpp = c_gp - c_p
        
        r_ideal = np.linalg.norm(v_hp)
        cos_theta = np.dot(v_hp, v_gpp) / (r_ideal * np.linalg.norm(v_gpp))
        theta_ideal = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        
        rotor_table["target_idx"].append(h_idx)
        rotor_table["parent_idx"].append(af3_lookup[p_key])
        rotor_table["grandparent_idx"].append(af3_lookup[gp_key])
        rotor_table["greatgrand_idx"].append(af3_lookup[ggp_key])
        rotor_table["ideal_r"].append(r_ideal)
        rotor_table["ideal_theta"].append(theta_ideal)
        
    logging.info(f"Successfully mapped {len(rotor_table['target_idx'])} kinematic rotors from template.")
    
    return {k: jnp.array(v, dtype=jnp.float32 if "ideal" in k else jnp.int32) for k, v in rotor_table.items()}

def _empty_rotor_table():
    return {
        "target_idx": jnp.array([], dtype=jnp.int32),
        "parent_idx": jnp.array([], dtype=jnp.int32),
        "grandparent_idx": jnp.array([], dtype=jnp.int32),
        "greatgrand_idx": jnp.array([], dtype=jnp.int32),
        "ideal_r": jnp.array([], dtype=jnp.float32),
        "ideal_theta": jnp.array([], dtype=jnp.float32)
    }
