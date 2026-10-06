"""Exports `MinimalRoseTTAFoldModule` to a single ONNX file.

The point of this is distribution: an `.onnx` file plus `pip install onnxruntime`
runs the model with no PyTorch, no CUDA toolkit and no compiler on the target
machine. See `COMPILING.md` for the full picture and for the alternatives that
were rejected (AOTInductor still links libtorch; ExecuTorch's CUDA support is
immature).

Only the dynamo exporter is used. The legacy TorchScript exporter
(`dynamo=False`) is deprecated, and on this model it is actively dangerous: it
produced a 266 MB file that loaded, ran, and returned values unrelated to the
real answer -- `index_add_` with duplicate indices is not supported and it did not
say so. Always run `verify_onnx_model.py` on whatever comes out of here.

Prerequisites, none of which are the RFdiffusion env:
    conda create -n crimm-export python=3.11
    conda activate crimm-export
    pip install torch numpy scipy omegaconf onnx onnxruntime onnxscript

`L` (sequence length) is exported as a symbolic dimension by default, so one
file serves every protein length; `--static` restores the old baked-in L.

Usage:
    python tools/export/export_onnx.py --out models/rf_loop.onnx
    python tools/export/verify_onnx_model.py models/rf_loop.onnx

Expect roughly 90 minutes with a dynamic L (45 with --static) and ~8 GB of RAM. The result is two files -- the graph
and a `.data` sidecar holding the weights, which the dynamo exporter splits out
to stay under protobuf's 2 GB message limit. **Both must ship together**, in the
same directory; ONNX Runtime resolves the sidecar by the name recorded in the
graph.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

from _paths import CKPT, MODELS, TEST_PDB  # noqa: E402  (also puts src/ on sys.path)
warnings.filterwarnings("ignore")

import torch

# The graph's input/output names; the sampler's ONNX wrapper relies on them.
from crimm_ml.models.base import INPUT_NAMES, OUTPUT_NAMES  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(MODELS / "rf_loop.onnx"))
    ap.add_argument("--checkpoint", default=str(CKPT))
    ap.add_argument("--pdb", default=str(TEST_PDB))
    ap.add_argument("--contig", default="A1-20/6/A30-79")
    ap.add_argument("--opset", type=int, default=18,
                    help="18 is the minimum: ScatterElements with reduction='max', "
                         "which the segment softmax needs.")
    ap.add_argument("--static", action="store_true",
                    help="bake the traced L into the graph (the old behaviour). "
                         "The default exports L as a symbolic dimension so one "
                         "file serves every protein length.")
    ap.add_argument("--lmin", type=int, default=8,
                    help="lower bound on the symbolic L. Mostly advisory: ONNX "
                         "records only the symbol name, not the range, and ORT "
                         "does not enforce it. An earlier default of 64 was set "
                         "defensively because `make_topk_graph` does "
                         "`min(top_k, L)` with top_k=64 -- but the exporter "
                         "lowers that to a real dynamic Min<-Shape, and a model "
                         "exported with min=64 runs correctly at L=27.")
    ap.add_argument("--lmax", type=int, default=2048)
    args = ap.parse_args()

    import crimm_ml.minimal_rf.util_module as UM
    # An exported artifact cannot contain a graph break, and torch.export can
    # represent `nonzero` on its own. See util_module.set_export_mode.
    UM.set_export_mode(True)
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._dynamo.config.capture_scalar_outputs = True

    from crimm_ml.models.onnx_model import embed_config
    from crimm_ml.models.torch_model import TorchRFModel, first_step_inputs

    log(f"torch {torch.__version__}")
    log("building the model and one real set of inputs ...")
    # Real inputs rather than zeros: tracing follows data-dependent paths
    # (top-k neighbour selection, the `nonzero` edge list), and all-zero
    # coordinates would exercise degenerate ones. The self-conditioning
    # template is zeroed, as at the first reverse-diffusion step.
    rf = TorchRFModel(args.checkpoint, device="cpu")
    model = rf.module
    _, inputs = first_step_inputs(open(args.pdb).readlines(), args.contig, rf)
    mm = inputs[0]

    # Which axis of each input is the sequence length, in `_preprocess` order:
    #   msa_latent (1,1,L,48)  msa_full (1,1,L,25)  seq (1,L,22)
    #   xyz (1,L,27,3)         idx (1,L)            t1d (1,1,L,22)
    #   t2d (1,1,L,L,44)       xyz_t (1,1,L,27,3)   alpha_t (1,1,L,30)
    # t2d is the only input where L appears twice, and both axes are the SAME
    # symbol -- it is a pair representation, so a (L, L') rectangle is not a
    # thing the model can consume.
    L_AXES = [(2,), (2,), (1,), (1,), (1,), (2,), (2, 3), (2,), (2,)]

    dynamic_shapes = None
    if not args.static:
        L = torch.export.Dim("L", min=args.lmin, max=args.lmax)
        dynamic_shapes = tuple({ax: L for ax in axes} for axes in L_AXES)

    mode = "static" if args.static else f"dynamic L in [{args.lmin}, {args.lmax}]"
    log(f"traced at L={mm.shape[2]}, exporting {mode} to {args.out} "
        f"(this takes ~45 min) ...")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.onnx.export(model, inputs, str(out), dynamo=True, opset_version=args.opset,
                      input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                      dynamic_shapes=dynamic_shapes, verbose=False)
    export_s = time.time() - t0
    log(f"EXPORT TIME: {export_s:.0f}s ({export_s/60:.1f} min)")
    # The sampler reads the checkpoint's diffusion/preprocessing settings from
    # the model; store them in the file so the ONNX model needs no checkpoint.
    embed_config(out, rf.config)
    log("embedded the checkpoint config in the model metadata")
    for f in sorted(out.parent.glob(out.name + "*")):
        log(f"  {f.name}  {f.stat().st_size / 1e6:.0f} MB")
    log("NOW VERIFY IT -- an export that succeeds is not necessarily correct:")
    log(f"  python tools/export/verify_onnx_model.py {out}")


if __name__ == "__main__":
    main()
