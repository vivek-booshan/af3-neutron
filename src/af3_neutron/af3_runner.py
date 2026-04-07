import functools
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import pathlib
from collections.abc import Callable
import tokamax

from alphafold3.model import model, features, params
from alphafold3.model.components import utils

def make_model_config(
    *,
    flash_attention_implementation: tokamax.DotProductAttentionImplementation = 'triton',
    num_diffusion_samples: int = 1, # Changed to 1 to speed up testing
    num_recycles: int = 10,
    return_embeddings: bool = False,
    return_distogram: bool = False,
) -> model.Model.Config:
    """Returns a model config with defaults overridden for neutron refinement."""
    config = model.Model.Config()
    config.global_config.flash_attention_implementation = flash_attention_implementation
    config.heads.diffusion.eval.num_samples = num_diffusion_samples
    config.num_recycles = num_recycles
    config.return_embeddings = return_embeddings
    config.return_distogram = return_distogram
    return config


class ModelRunner:
    """Helper class to run AF3 structure prediction stages."""

    def __init__(self, config: model.Model.Config, device: jax.Device, model_dir: pathlib.Path):
        self._model_config = config
        self._device = device
        self._model_dir = model_dir

    @functools.cached_property
    def model_params(self) -> hk.Params:
        """Loads model parameters from the model directory."""
        return params.get_model_haiku_params(model_dir=self._model_dir)

    @functools.cached_property
    def _model(self) -> Callable[[jnp.ndarray, features.BatchDict], model.ModelResult]:
        """Loads model parameters and returns a jitted model forward pass."""
        @hk.transform
        def forward_fn(batch):
            return model.Model(self._model_config)(batch)

        return functools.partial(
            jax.jit(forward_fn.apply, device=self._device), self.model_params
        )

    def run_inference(self, featurised_example: features.BatchDict, rng_key: jnp.ndarray) -> model.ModelResult:
        """Computes a forward pass of the model on a featurised example."""
        featurised_example = jax.device_put(
            jax.tree_util.tree_map(
                jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
            ),
            self._device,
        )

        result = self._model(rng_key, featurised_example)
        result = jax.tree.map(np.asarray, result)
        result = jax.tree.map(
            lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
            result,
        )
        result = dict(result)
        identifier = self.model_params['__meta__']['__identifier__'].tobytes()
        result['__identifier__'] = identifier
        return result

    def extract_inference_results(self, batch: features.BatchDict, result: model.ModelResult, target_name: str):
        """Extracts inference results from model outputs."""
        return list(
            model.Model.get_inference_result(
                batch=batch, result=result, target_name=target_name
            )
        )
