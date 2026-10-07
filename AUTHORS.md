# Authors

Who wrote crimm-ml. The rules for this file are in
[CONTRIBUTING.md](CONTRIBUTING.md#credit-and-authorship); third-party code this project is
derived from is credited in [NOTICE](NOTICE), not here.

## People

- **Truman Xu (徐梓乔)** — owner and maintainer. Conceived and directed the project, set its
  requirements and design constraints, reviewed and tested the work, and wrote the
  residue-rebuilding postprocessing that crimm's loop builder is based on.

## AI models

One entry per model family. The individual changes, and the exact model version that made
each, are in the `Co-Authored-By` trailers of the git history.

- **Claude** (Anthropic) — since the first commit. Wrote the code in this repository under
  the owner's direction:

  | Area | Contribution |
  | --- | --- |
  | `src/crimm_ml/numpy_rf/`, `minimal_loop_sampler.py` | NumPy port of RFdiffusion's diffusion, denoising and sampling code, with the equivalence tests against the PyTorch original |
  | `src/crimm_ml/minimal_rf/`, `minimal_rf_module.py`, `minimal_track_module.py` | Reduction of RFdiffusion's network and inference code to the loop-building case, and the changes that made it exportable with a dynamic sequence length |
  | `src/crimm_ml/minimal_rf/se3/` | Rewrite of the SE(3)-Transformer without DGL or e3nn |
  | `src/crimm_ml/models/`, `assets.py`, `cli.py`, `errors.py`, `progress.py` | Model loading for ONNX Runtime and PyTorch, model fetching and verification, command line, progress reporting |
  | `src/crimm_ml/compilation.py`, `compiled_model.py` | `torch.compile` support and its portable cache |
  | `src/crimm_ml/crimm_utils.py`, `minimal_pipeline_loop_builder.py` | The first crimm-facing glue (deprecated) |
  | `tools/` | ONNX export, the ONNX Runtime graph rewrites, verification, benchmarking and profiling tools |
  | `tests/`, `docs/` | Test suite, golden fixtures, documentation and decision records |
