"""Times a dynamic-L ONNX export and checks it is still correct at each length.

Answers three questions, in the order they matter:

  1. Does one file really serve every L?  The export declares L symbolic, but a
     declaration is not evidence -- an exporter can still bake a dimension in.
     This runs several lengths through the SAME session and compares each
     against torch.
  2. What does it cost to load?  Session creation is the startup cost the whole
     ONNX route exists to minimise, so it is timed separately from inference
     and reported cold (first session in the process).
  3. What does it cost to run?  First call and steady-state median are reported
     separately -- ORT does shape-dependent setup on the first call at each new
     shape, which would otherwise be charged to inference.

Usage:
    python tools/export/bench_onnx.py models/rf_loop_dynamic.onnx
    python tools/export/bench_onnx.py <model> --threads 16 --repeats 5
"""
import argparse
import functools
import statistics
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

#: Only these three are read by `MinimalLoopSampler._sample_step`; the rest are
#: internal activations whose absolute scale is arbitrary. Same bar the ONNX
#: verifier uses.
CONSUMED = {"xyz_out": 1e-2, "alpha": 1e-2, "pred_lddt": 1e-2}

#: 2KL8 chain A. Only the diffused-segment length and the right-hand island
#: change, so these differ in L and nothing else. 57 is below `make_topk_graph`'s
#: top_k=64, so it exercises the `min(top_k, L)` branch.
CONTIGS = [("A1-20/6/A30-79", 76), ("A1-20/10/A30-79", 80),
           ("A1-20/30/A30-79", 100), ("A1-20/6/A30-60", 57)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@functools.lru_cache(maxsize=None)
def torch_reference(device="cpu", checkpoint=str(CKPT)):
    """The PyTorch model, loaded once per device."""
    from crimm_ml.models.torch_model import TorchRFModel
    return TorchRFModel(checkpoint, device=device)


def build_case(pdb_lines, contig, device="cpu"):
    """The torch module plus one real input set, as the sampler feeds it at the
    first reverse-diffusion step (self-conditioning template all zeros)."""
    import crimm_ml.minimal_rf.util_module as UM
    UM.set_export_mode(True)   # the export has no fence; neither should the reference
    from crimm_ml.models.torch_model import first_step_inputs

    model = torch_reference(device)
    _, inputs = first_step_inputs(pdb_lines, contig, model)
    return model.module, inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--pdb", default=str(TEST_PDB))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3, help="timed steady-state calls per L")
    ap.add_argument("--skip-torch", action="store_true",
                    help="timing only -- do not build the torch reference")
    args = ap.parse_args()

    import onnxruntime as ort
    torch.set_num_threads(args.threads)

    path = Path(args.model)
    files = sorted(path.parent.glob(path.name + "*"))
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    log(f"onnxruntime {ort.__version__}  torch {torch.__version__}  "
        f"threads={args.threads}")
    for f in files:
        log(f"  {f.name}  {f.stat().st_size / 1e6:.0f} MB")

    # ---- LOAD -------------------------------------------------------------
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    t0 = time.time()
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    load_s = time.time() - t0
    log(f"LOAD TIME: {load_s:.2f}s  (session creation, {total_mb:.0f} MB on disk)")

    declared = {i.name: i.shape for i in sess.get_inputs()}
    log("declared input shapes (a string entry means that axis is symbolic):")
    for n in INPUT_NAMES:
        log(f"  {n:12s} {declared.get(n)}")

    pdb_lines = open(args.pdb).readlines()
    rows = []
    for contig, expect_L in CONTIGS:
        model, inputs = build_case(pdb_lines, contig)
        L = inputs[0].shape[2]
        assert L == expect_L, f"{contig} gave L={L}"
        feed = {n: t.detach().cpu().numpy() for n, t in zip(INPUT_NAMES, inputs)}

        log(f"--- L={L} ({contig}) ---")
        try:
            t0 = time.time(); got = sess.run(None, feed); first_s = time.time() - t0
        except Exception as e:
            msg = str(e).split("\n")[0][:140]
            log(f"  REFUSED: {type(e).__name__}: {msg}")
            rows.append((L, None, None, None, "refused"))
            continue

        times = []
        for _ in range(args.repeats):
            t0 = time.time(); sess.run(None, feed); times.append(time.time() - t0)
        onnx_s = statistics.median(times)

        verdict, torch_s = "ok", None
        if not args.skip_torch:
            t0 = time.time()
            with torch.no_grad():
                ref = model(*inputs)
            torch_s = time.time() - t0
            worst = []
            for name, r, g in zip(OUTPUT_NAMES, ref, got):
                d = float(np.abs(r.detach().cpu().numpy() - g).max())
                if name in CONSUMED:
                    worst.append((name, d, d <= CONSUMED[name]))
            for name, d, ok in worst:
                log(f"  {name:10s} max|diff vs torch| = {d:.3e}  "
                    f"{'OK' if ok else 'FAIL'}")
            verdict = "ok" if all(ok for _, _, ok in worst) else "MISMATCH"
        log(f"  first call {first_s:.2f}s   steady {onnx_s:.2f}s"
            + (f"   torch eager {torch_s:.2f}s" if torch_s else ""))
        rows.append((L, first_s, onnx_s, torch_s, verdict))

    print()
    print("=" * 74)
    print(f"model      : {path.name}  ({total_mb:.0f} MB)")
    print(f"LOAD TIME  : {load_s:.2f}s")
    print("=" * 74)
    print(f"{'L':>5} {'first call':>11} {'steady':>9} {'torch eager':>12} "
          f"{'speedup':>8}  {'correctness':>11}")
    for L, first_s, onnx_s, torch_s, verdict in rows:
        if first_s is None:
            print(f"{L:>5} {'-':>11} {'-':>9} {'-':>12} {'-':>8}  {verdict:>11}")
            continue
        sp = f"{torch_s / onnx_s:.2f}x" if torch_s else "-"
        print(f"{L:>5} {first_s:>10.2f}s {onnx_s:>8.2f}s "
              f"{(f'{torch_s:.2f}s' if torch_s else '-'):>12} {sp:>8}  {verdict:>11}")


if __name__ == "__main__":
    main()
