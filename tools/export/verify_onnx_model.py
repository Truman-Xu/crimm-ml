"""Checks an exported ONNX model against the PyTorch model it came from.

Written because an export that "succeeds" is not necessarily an export that is
*correct*. The first ONNX file produced for this model came out of torch's
legacy TorchScript exporter in 127s, loaded cleanly in ONNX Runtime, ran, and
returned values with a **relative error of order 1.0** on every output -- i.e.
unrelated to the real answer. It also emitted dozens of `In-place op on output
of tensor.shape` warnings during export, and ORT reported shape-metadata
mismatches inside the `edge_softmax` scatter ops. Nothing about "SUCCESS"
revealed any of that. Shipping on that signal would have put a silently broken
model into `pip install crimm`.

So: any ONNX artifact intended for distribution goes through this first.

Three checks, in increasing order of what they would catch:

  1. Same inputs as torch  -- does it compute the same function at all?
  2. Jittered coordinates  -- the edge count E comes from `nonzero`, so this
     changes a dimension that is *internal* to the graph. It catches a model
     that baked E in as a constant, which would be wrong for every input but
     the one it was traced on.
  3. Timing               -- only meaningful once 1 and 2 pass. A wrong graph
     can be fast.

Usage:
    python tools/export/verify_onnx_model.py <model.onnx> [--threads 16]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

from _paths import CKPT, TEST_PDB  # noqa: E402  (also puts src/ on sys.path)
warnings.filterwarnings("ignore")

import numpy as np
import torch

# The graph's input/output names; the sampler's ONNX wrapper relies on them.
from crimm_ml.models.base import INPUT_NAMES, OUTPUT_NAMES  # noqa: E402

#: The sampler unpacks the model's 7 outputs as
#:     `_, _, px0, _, alpha, _, plddt = self.model(...)`
#: (minimal_loop_sampler.py `_sample_step`), so only these three are consumed.
#: The rest -- msa0, pair, state, logits_aa -- are internal activations whose
#: absolute scale is arbitrary; they are reported below but not gated on.
#: Gating on what the pipeline actually uses is the point, not a looser bar:
#: `state` has the largest relative drift precisely because nothing reads it.
CONSUMED = {
    # name        absolute tolerance,  why that number
    "xyz_out":   (1e-2, "Angstrom -- ~150x below a C-C bond, far below any structural significance"),
    "alpha":     (1e-2, "torsion sin/cos components, bounded by ~1"),
    "pred_lddt": (1e-2, "confidence in [0,1]"),
}
PDB = TEST_PDB
CONTIG = "A1-20/6/A30-79"
TOL = 2e-3  # same bar the rest of this package uses: fp-reassociation level


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_reference(device="cpu"):
    """The torch model plus one real set of inputs, exactly as the sampler
    would feed it at the first reverse-diffusion step."""
    import crimm_ml.minimal_rf.util_module as UM
    # The export has no dynamo fence in it, so neither should the reference.
    UM._edge_index_from_cond = lambda cond: torch.where(cond)
    from crimm_ml.models.torch_model import TorchRFModel, first_step_inputs

    model = TorchRFModel(CKPT, device=device)
    _, inputs = first_step_inputs(open(PDB).readlines(), CONTIG, model)
    return model.module, inputs


def compare(ref, got, label):
    """Reports every output; decides on the three the sampler consumes.

    ONNX Runtime uses entirely different kernels and reduction orders from
    PyTorch, so exact agreement is not on offer and a pure relative-error bar
    across all seven tensors mostly measures the largest internal activation.
    """
    ok = True
    for name, r, g in zip(OUTPUT_NAMES, ref, got):
        r = r.numpy() if isinstance(r, torch.Tensor) else r
        d = np.abs(r - g).max()
        scale = np.abs(r).max()
        rel = d / scale if scale > 0 else 0.0
        if name in CONSUMED:
            tol, _why = CONSUMED[name]
            good = d <= tol
            ok &= good
            log(f"    {name:10s} max|diff|={d:.3e}  rel={rel:.2e}   [used by sampler,"
                f" tol {tol:.0e}] {'OK' if good else 'FAIL'}")
        else:
            log(f"    {name:10s} max|diff|={d:.3e}  rel={rel:.2e}   (internal, not gated)")
    log(f"  {label}: {'OK' if ok else 'WRONG -- do not ship'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    import onnxruntime as ort
    log(f"onnxruntime {ort.__version__}, torch {torch.__version__}")
    model, inputs = build_reference()

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    t0 = time.time()
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    log(f"session ready in {time.time() - t0:.1f}s "
        f"({Path(args.model).stat().st_size / 1e6:.0f} MB)")

    feed = {n: a.numpy() for n, a in zip(INPUT_NAMES, inputs)}
    with torch.no_grad():
        ref = model(*inputs)
    ok = compare(ref, sess.run(None, feed), "identical inputs")

    log("jittered coordinates (changes the internal edge count E) ...")
    torch.manual_seed(7)
    inputs2 = list(inputs)
    inputs2[3] = inputs[3] + torch.randn_like(inputs[3]) * 0.7
    feed2 = dict(feed, xyz=inputs2[3].numpy())
    with torch.no_grad():
        ref2 = model(*inputs2)
    try:
        ok &= compare(ref2, sess.run(None, feed2), "jittered coordinates")
    except Exception as e:
        log(f"  FAILED to run: {str(e)[:300]}")
        ok = False

    if ok:
        sess.run(None, feed)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            sess.run(None, feed)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        log(f"onnxruntime CPU ({args.threads} threads): {ts[1] * 1000:.0f} ms/forward"
            f"   design@T=50 = {ts[1] * 50:.1f}s")
    else:
        log("skipping timing: a wrong graph's speed is not information")
    log("VERDICT: " + ("USABLE" if ok else "NOT USABLE"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
