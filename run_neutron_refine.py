import logging
import pathlib
import jax
import jax.numpy as jnp
from absl import app
from absl import flags

from alphafold3.common import folding_input
from alphafold3.data import featurisation
from alphafold3.constants import chemical_components

# Use our lifted local runner
from af3_neutron.af3_runner import ModelRunner, make_model_config

from af3_neutron.topology import build_oracle_from_af3_result
from af3_neutron.sampler import run_neutron_guided_diffusion

FLAGS = flags.FLAGS
flags.DEFINE_string('json_path', 'betalac_tetramer_refinement_input.json', 'Path to JSON.')
flags.DEFINE_string('model_dir', '../af3_model_parameters/', 'Path to weights.')
flags.DEFINE_integer('gpu_device', 0, 'GPU to use.')

def main(argv):
    del argv
    logging.basicConfig(level=logging.INFO)
    
    json_path = pathlib.Path(FLAGS.json_path)
    model_dir = pathlib.Path(FLAGS.model_dir)
    
    # 1. Load Input
    fold_inputs = folding_input.load_fold_inputs_from_path(json_path)
    fold_input = next(fold_inputs)
    ccd = chemical_components.Ccd()
    
    # 2. Native AF3 Featurisation (Generates MSAs and 24-atom tokens properly!)
    logging.info("Running native AF3 featurisation...")
    # NOTE: If your JSON does not already contain MSAs, this will trigger the HMMER searches.
    # Ensure your paths to DB_DIR in run_alphafold are configured if starting from scratch.
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input,
        buckets=[256, 512, 1024, 2048, 4096],
        ccd=ccd,
        verbose=True
    )
    
    # 3. Run Native AF3 Inference
    logging.info("Loading AF3 Weights...")
    config = make_model_config()
    device = jax.local_devices(backend='gpu')[FLAGS.gpu_device]
    
    model_runner = ModelRunner(config=config, device=device, model_dir=model_dir)
    
    logging.info("Executing AlphaFold 3 Neural Network...")
    rng_key = jax.random.PRNGKey(42)
    result = model_runner.run_inference(featurised_examples[0], rng_key)
    
    # Extract the final output object
    inference_results = model_runner.extract_inference_results(
        batch=featurised_examples[0], result=result, target_name=fold_input.name
    )
    af3_final_state = inference_results[0]
    
    # 4. State Augmentation (The Crystallographic Oracle)
    rotor_table, mapping = build_oracle_from_af3_result(af3_final_state)
    
    # Extract predicted coordinates from metadata
    initial_af3_coords = []
    for atom in af3_final_state.metadata['atom_pos']:
        initial_af3_coords.append(atom) # Shape (N_atoms, 3)
        
    # 5. Run Decoupled JAX Refinement
    final_coords, final_chis = run_neutron_guided_diffusion(
        initial_af3_coords=initial_af3_coords, 
        rotor_table=rotor_table, 
        mapping=mapping
    )
    
    logging.info("Neutron Refinement Complete.")

if __name__ == "__main__":
    app.run(main)
