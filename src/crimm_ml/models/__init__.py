"""The network behind `MinimalLoopSampler`, loaded as a PyTorch or an ONNX model.

    from crimm_ml.models import load_model
    from crimm_ml.minimal_loop_sampler import MinimalLoopSampler

    model = load_model()                                       # the cached ONNX model
    model = load_model("models/rf_loop_dynamic_tp_ort.onnx")   # an ONNX file
    model = load_model("models/Base_ckpt.pt", device="cuda")   # PyTorch
    sampler = MinimalLoopSampler(pdb_lines, contig, model, T=50)

Both kinds are called the same way -- NumPy arrays in, NumPy arrays out -- and
carry `config`, the checkpoint settings the sampler needs. A model can be
loaded once and shared by any number of samplers.

Importing this package does not import torch, and neither does loading or
running an ONNX model; only `TorchRFModel` does.
"""
from pathlib import Path

from .base import CONFIG_METADATA_KEY, CONFIG_SECTIONS, INPUT_NAMES, OUTPUT_NAMES, RFModel

TORCH_SUFFIXES = (".pt", ".pth", ".ckpt")

__all__ = ["load_model", "RFModel", "INPUT_NAMES", "OUTPUT_NAMES", "CONFIG_METADATA_KEY",
           "CONFIG_SECTIONS"]


def load_model(path=None, device=None, threads=None):
    """Load the network from `path`, choosing the backend by file suffix.

    No `path`: the ONNX model in crimm_ml's cache (`crimm_ml.assets`),
    fetched first if it is not there yet.

    `.onnx` -> `onnx_model.OnnxRFModel` (ONNX Runtime; torch is never
    imported). The file must carry the checkpoint config in its metadata
    (`examples/onnx_embed_config.py` adds it to older exports).
    `.pt` / `.pth` / `.ckpt` -> `torch_model.TorchRFModel` (a checkpoint
    such as `Base_ckpt.pt`).

    `device`: `"cpu"`, `"cuda"` or `"cuda:N"`. `None` picks CUDA when it is
    available to the chosen backend, else CPU.
    `threads`: ONNX Runtime's intra-op thread count (ONNX only). `None` keeps
    ORT's default of one thread per physical core; set it on a shared machine.
    """
    if path is None:
        from crimm_ml import assets
        path = assets.model_path(fetch_if_missing=True)
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        from .onnx_model import OnnxRFModel
        return OnnxRFModel(path, device=device, threads=threads)
    if suffix in TORCH_SUFFIXES:
        if threads is not None:
            raise ValueError("threads= applies to ONNX models; for PyTorch use torch.set_num_threads()")
        from .torch_model import TorchRFModel
        return TorchRFModel(path, device=device)
    raise ValueError(f"don't know how to load {path.name!r}: expected .onnx or one of {TORCH_SUFFIXES}")
