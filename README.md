# crimm-ml

RFdiffusion loop building for [crimm](https://github.com/BrooksResearchGroup-UM/crimm),
running on ONNX Runtime — **no PyTorch, no CUDA toolkit, no compiler**.

```bash
pip install crimm[ml]        # or: pip install crimm-ml
```

The runtime is numpy, scipy, omegaconf and onnxruntime. On CPU the ONNX model
runs at PyTorch-eager speed (0.99 s against 0.90 s per forward pass at 76
residues, 8 threads); on a GPU it is about twice as fast as PyTorch. The
sampler around it is NumPy, so the whole reverse-diffusion loop runs without
torch installed.

## Quick start

```python
from crimm_ml import load_model, MinimalLoopSampler

model = load_model("rf_loop_dynamic_tp_ort.onnx", threads=8)   # load once, reuse
pdb_lines = open("2KL8.pdb").readlines()

# keep residues A1-20 and A27-79, build a 6-residue loop between them
sampler = MinimalLoopSampler(pdb_lines, "A1-20/6-6/A27-79", model, T=50)
design = sampler.sample(num_designs=1, seed=0)[0]
design["x_final"]        # (L, 14, 3) coordinates, the new loop included
design["plddt_final"]    # per-residue confidence
```

Filling the gaps of a crimm chain — choosing gaps, writing the contig and
rebuilding complete residues from the result — lives in crimm itself, which
calls this package.

`sample(progress=...)` shows a tqdm bar in notebooks and terminals when tqdm is
installed (`pip install crimm-ml[progress]`), or calls a function you pass with
`(done, total, info)` after every step.

## The model file

The 275 MB ONNX model (`.onnx` plus its `.onnx.data` weights, which must stay
in the same directory) is too large for PyPI and is downloaded separately.
Cached files live under `$CRIMM_ML_CACHE`, or the platform cache directory
(`~/.cache/crimm-ml` on Linux).

The IGSO(3) schedule table for each number of steps `T` is built on first use
(about a minute) and cached there too.

## Running the original PyTorch checkpoint

`load_model("Base_ckpt.pt")` runs RFdiffusion's checkpoint through the same
sampler, after `pip install crimm-ml[torch]`. Results agree with the ONNX
model to a few millionths of an ångström.

## Building and verifying the ONNX model

`tools/export/` re-exports the model from the PyTorch checkpoint, applies the
ONNX Runtime graph rewrites, and verifies the result against PyTorch. These
tools need `pip install -e .[export]` and an RFdiffusion checkout with
`models/Base_ckpt.pt` (set `RFDIFFUSION_DIR`). `docs/COMPILING.md` covers the
whole procedure and the measurements; `docs/DISTRIBUTING.md` explains why the
ONNX export is the shipped form.

## Licence

GPL-3.0-or-later. Portions are derived from RFdiffusion (BSD licence, which
also covers the model weights); see `NOTICE`.
