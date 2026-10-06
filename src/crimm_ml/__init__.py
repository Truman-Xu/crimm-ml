"""crimm-ml: RFdiffusion loop building for crimm, without PyTorch.

    from crimm_ml import load_model, MinimalLoopSampler

    model = load_model("rf_loop_dynamic_tp_ort.onnx")      # ONNX Runtime
    sampler = MinimalLoopSampler(pdb_lines, "A1-20/6-6/A27-79", model, T=50)
    result = sampler.sample(num_designs=1)[0]               # result["x_final"]: (L, 14, 3)

The network is a separate object from `load_model`: an ONNX export run by ONNX
Runtime, or a PyTorch checkpoint (needs the `[torch]` extra). Around it the
sampler is NumPy (`numpy_rf`), so with an ONNX model nothing imports torch.
`sample(progress=...)` reports the diffusion steps -- a tqdm bar when tqdm is
installed and the output is interactive, or a callback of your own (see
`progress`).

Building loops into a crimm chain -- choosing gaps, writing the contig,
rebuilding full residues from the result -- belongs to crimm itself, which
calls this package. `crimm_utils` and `minimal_pipeline_loop_builder` are the
earlier versions of that glue, kept for one release and deprecated.

The PyTorch-only modules (`minimal_rf_module`, `compilation`,
`compiled_model`) are imported only when used. See `docs/` for how the ONNX
model is built and verified, and why it is the shipped form.
"""
import logging
from importlib.metadata import PackageNotFoundError, version as _version

from .errors import AssetError, ContigError, CrimmMLError
from .minimal_loop_sampler import MinimalLoopSampler
from .models import load_model

try:
    __version__ = _version("crimm-ml")
except PackageNotFoundError:          # running from a source tree, not installed
    __version__ = "0+unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ["load_model", "MinimalLoopSampler", "CrimmMLError", "ContigError", "AssetError",
           "__version__"]
