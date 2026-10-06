"""The network in ONNX Runtime. No torch anywhere on this path.

The file comes from `examples/export_onnx.py`, usually followed by
`examples/onnx_ort_rewrites.py`; its `.data` weights file must sit next to it
under the name recorded at export. The checkpoint settings the sampler needs
are stored in the file's metadata under `CONFIG_METADATA_KEY`
(`embed_config` below writes them). See COMPILING.md section 4.
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np

from .base import CONFIG_METADATA_KEY, CONFIG_SECTIONS, INPUT_NAMES, OUTPUT_NAMES, RFModel

_ORT_TYPES = {"tensor(float)": np.float32, "tensor(int64)": np.int64}


class OnnxRFModel(RFModel):
    """An ONNX Runtime session with the calling convention of `RFModel`.

    `device`: `"cpu"`, `"cuda"` or `"cuda:N"`; `None` picks CUDA if this
    `onnxruntime` has the CUDA provider. With CUDA, the CUDA 12 / cuDNN 9
    libraries are preloaded (`onnxruntime.preload_dlls`), and if ORT still
    ends up on CPU -- it does not raise when a provider fails to load -- this
    warns and `device` says `"cpu"`.
    `threads`: ORT's intra-op thread count; `None` keeps ORT's default.
    """

    backend = "onnx"

    def __init__(self, path, device=None, threads=None):
        import onnxruntime as ort

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.path}")

        available = ort.get_available_providers()
        if device is None:
            device = "cuda" if "CUDAExecutionProvider" in available else "cpu"
        kind, _, index = str(device).partition(":")
        if kind not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu', 'cuda' or 'cuda:N', got {device!r}")

        providers = ["CPUExecutionProvider"]
        if kind == "cuda":
            if "CUDAExecutionProvider" in available:
                providers.insert(0, ("CUDAExecutionProvider", {"device_id": int(index or 0)}))
                if hasattr(ort, "preload_dlls"):
                    try:
                        ort.preload_dlls()
                    except Exception:  # best effort; the provider check below reports failure
                        pass
            else:
                warnings.warn("device is CUDA but this onnxruntime has no CUDA provider "
                              "(install onnxruntime-gpu); running the ONNX model on CPU")

        so = ort.SessionOptions()
        if threads is not None:
            so.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(str(self.path), so, providers=providers)
        self.providers = self.session.get_providers()
        on_cuda = "CUDAExecutionProvider" in self.providers
        if kind == "cuda" and not on_cuda and "CUDAExecutionProvider" in available:
            warnings.warn(f"requested CUDA but ONNX Runtime is using {self.providers}; "
                          "see its error log above for the library it could not load")
        self.device = f"cuda:{int(index or 0)}" if on_cuda else "cpu"

        inputs = self.session.get_inputs()
        got_in = [i.name for i in inputs]
        got_out = [o.name for o in self.session.get_outputs()]
        if got_in != INPUT_NAMES or got_out != OUTPUT_NAMES:
            raise ValueError(
                f"{self.path} does not look like an export of this model: inputs {got_in} "
                f"(expected {INPUT_NAMES}), outputs {got_out} (expected {OUTPUT_NAMES})")
        self._input_dtypes = [_ORT_TYPES.get(i.type) for i in inputs]

        meta = self.session.get_modelmeta().custom_metadata_map
        if CONFIG_METADATA_KEY not in meta:
            raise ValueError(
                f"{self.path} carries no model config. Add it once with\n"
                f"    python tools/export/onnx_embed_config.py {self.path}")
        self.config = json.loads(meta[CONFIG_METADATA_KEY])

    def __call__(self, *inputs):
        if len(inputs) != len(INPUT_NAMES):
            raise TypeError(f"expected {len(INPUT_NAMES)} inputs, got {len(inputs)}")
        feed = {}
        for name, want, a in zip(INPUT_NAMES, self._input_dtypes, inputs):
            a = np.ascontiguousarray(a)
            if want is not None and a.dtype != want:
                raise TypeError(f"input {name!r} is {a.dtype}, the model expects {np.dtype(want)}")
            feed[name] = a
        return tuple(self.session.run(OUTPUT_NAMES, feed))


def embed_config(onnx_path, config):
    """Store `config` (a dict with `CONFIG_SECTIONS`) in the metadata of the
    ONNX file at `onnx_path`, in place. Rewrites the graph file only (the
    weights file is untouched), via a temporary file and an atomic rename.
    Needs the `onnx` package, which running a model does not."""
    import onnx

    onnx_path = str(onnx_path)
    value = json.dumps({k: dict(config[k]) for k in CONFIG_SECTIONS}, sort_keys=True)
    m = onnx.load(onnx_path, load_external_data=False)
    for p in m.metadata_props:
        if p.key == CONFIG_METADATA_KEY:
            p.value = value
            break
    else:
        m.metadata_props.add(key=CONFIG_METADATA_KEY, value=value)
    tmp = onnx_path + ".tmp"
    onnx.save_model(m, tmp)
    os.replace(tmp, onnx_path)
