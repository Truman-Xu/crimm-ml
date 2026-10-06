"""The network in PyTorch, from a checkpoint such as `Base_ckpt.pt`."""
from pathlib import Path

import numpy as np
import torch

from crimm_ml.minimal_rf_module import MinimalRoseTTAFoldModule

from .base import CONFIG_SECTIONS, RFModel


class TorchRFModel(RFModel):
    """`MinimalRoseTTAFoldModule` in eager mode, with NumPy in and out.

    `module` is the underlying `torch.nn.Module`, for anything that needs the
    torch object itself (exporting, compiling, profiling).
    """

    backend = "torch"

    def __init__(self, checkpoint_path, device=None):
        self.path = Path(checkpoint_path)
        self.torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(self.path, map_location=self.torch_device)
        self.config = {k: dict(ckpt["config_dict"][k]) for k in CONFIG_SECTIONS}
        self.module, self.load_result = MinimalRoseTTAFoldModule.from_checkpoint(
            ckpt, device=self.torch_device)
        self.device = str(self.torch_device)

    def __call__(self, *inputs):
        tensors = [torch.from_numpy(np.ascontiguousarray(a)).to(self.torch_device) for a in inputs]
        with torch.no_grad():
            outputs = self.module(*tensors)
        return tuple(o.detach().cpu().numpy() for o in outputs)


def first_step_inputs(pdb_lines, contig, model, T=15, seed=0, zero_template=True):
    """The nine network inputs `MinimalLoopSampler` builds for its first
    reverse-diffusion step, as torch tensors on `model`'s device -- for
    exporting, profiling and testing the network outside the sampler.

    With `zero_template=True` (the default) the self-conditioning template is
    zeroed as the sampler does at t = T (`xyz_t` and the first 44 channels of
    `t2d`). The initial noise is seeded with `seed`, so repeated calls give
    the same inputs. Returns `(sampler, inputs)`.
    """
    from crimm_ml.minimal_loop_sampler import MinimalLoopSampler

    sampler = MinimalLoopSampler(pdb_lines, contig, model, T=T)
    sampler._make_deterministic(seed)
    x_init, seq_init = sampler._sample_init()
    arrays = list(sampler._preprocess(seq_init, x_init, t=sampler.T))
    if zero_template:
        arrays[7] = np.zeros_like(arrays[7])      # xyz_t
        arrays[6] = arrays[6].copy()
        arrays[6][..., :44] = 0.0                 # t2d template channels
    inputs = tuple(torch.from_numpy(a).to(model.torch_device) for a in arrays)
    return sampler, inputs
