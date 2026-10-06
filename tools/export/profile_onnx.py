"""Where does ONNX Runtime actually spend its time, and where does torch?

`bench_onnx.py` says the ONNX model is ~2x slower than torch eager on CPU. This
answers *why* by measuring instead of arguing: ORT's own per-node profiler and
`torch.profiler` on the same inputs, both aggregated by operator type and put
side by side.

Read the two tables together. An op type near the top of the ORT column and
absent from the torch column is a lowering artifact -- work ONNX invented that
torch never does (shape arithmetic, materialized transposes). An op type at the
top of BOTH is a genuine kernel-quality gap.

Usage:
    python tools/export/profile_onnx.py models/rf_loop_dynamic.onnx
    python tools/export/profile_onnx.py <model> --contig A1-20/30/A30-79
"""
import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

from _paths import TEST_PDB  # noqa: E402  (also puts src/ on sys.path)
warnings.filterwarnings("ignore")

import numpy as np
import torch



def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def table(title, rows, total_ms, topn=18):
    """rows: list of (label, total_ms, count)."""
    print(f"\n{title}")
    print(f"  {'op type':24s} {'ms/run':>9} {'% time':>7} {'count':>8} {'us/call':>9}")
    print("  " + "-" * 62)
    for label, ms, n in rows[:topn]:
        pct = 100.0 * ms / total_ms if total_ms else 0.0
        print(f"  {label:24s} {ms:9.1f} {pct:7.1f} {n:8d} "
              f"{1000.0 * ms / n if n else 0:9.1f}")
    shown = sum(r[1] for r in rows[:topn])
    print(f"  {'(everything else)':24s} {total_ms - shown:9.1f} "
          f"{100.0 * (total_ms - shown) / total_ms if total_ms else 0:7.1f}")
    print(f"  {'TOTAL':24s} {total_ms:9.1f} {100.0:7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--pdb", default=str(TEST_PDB))
    ap.add_argument("--contig", default="A1-20/6/A30-79")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--runs", type=int, default=3, help="profiled iterations")
    args = ap.parse_args()

    import onnxruntime as ort
    from bench_onnx import build_case, INPUT_NAMES

    torch.set_num_threads(args.threads)
    log(f"onnxruntime {ort.__version__}  torch {torch.__version__}  "
        f"threads={args.threads}")

    model, inputs = build_case(open(args.pdb).readlines(), args.contig)
    L = inputs[0].shape[2]
    feed = {n: t.detach().cpu().numpy() for n, t in zip(INPUT_NAMES, inputs)}
    log(f"L={L} ({args.contig})")

    # ---------------- ONNX Runtime ----------------
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.enable_profiling = True
    so.profile_file_prefix = str(Path(args.model).parent / "ortprof")
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    sess.run(None, feed)                       # warm up; still recorded, divided out below
    for _ in range(args.runs):
        sess.run(None, feed)
    prof_path = sess.end_profiling()
    log(f"ORT profile -> {prof_path}")

    events = json.loads(Path(prof_path).read_text())
    by_op, by_node = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
    n_runs = 0
    for e in events:
        cat, dur = e.get("cat"), e.get("dur", 0)
        if cat == "Session" and e.get("name") == "model_run":
            n_runs += 1
        elif cat == "Node" and e.get("name", "").endswith("_kernel_time"):
            op = e.get("args", {}).get("op_name", "?")
            by_op[op][0] += dur / 1000.0
            by_op[op][1] += 1
            by_node[(e["name"][:-12], op)][0] += dur / 1000.0
            by_node[(e["name"][:-12], op)][1] += 1
    n_runs = max(n_runs, 1)
    log(f"profiled {n_runs} model_run events")

    ort_rows = sorted(((k, v[0] / n_runs, v[1] // n_runs) for k, v in by_op.items()),
                      key=lambda r: -r[1])
    ort_total = sum(r[1] for r in ort_rows)
    table(f"=== ONNX Runtime, per run, L={L} ===", ort_rows, ort_total)

    print("\n  hottest individual nodes:")
    node_rows = sorted(((k, v[0] / n_runs) for k, v in by_node.items()),
                       key=lambda r: -r[1])[:8]
    for (name, op), ms in node_rows:
        print(f"    {ms:8.1f} ms  {op:20s} {name[:44]}")

    # ---------------- torch eager ----------------
    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad():
        model(*inputs)                                     # warm up
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for _ in range(args.runs):
                model(*inputs)
    ev = prof.key_averages()
    t_rows = sorted(((e.key, e.self_cpu_time_total / 1000.0 / args.runs,
                      e.count // args.runs) for e in ev if e.self_cpu_time_total > 0),
                    key=lambda r: -r[1])
    t_total = sum(r[1] for r in t_rows)
    table(f"=== torch eager (self CPU time), per run, L={L} ===", t_rows, t_total)

    print(f"\n{'=' * 66}")
    print(f"ORT   total per run: {ort_total / 1000:6.2f}s")
    print(f"torch total per run: {t_total / 1000:6.2f}s")
    print(f"{'=' * 66}")


if __name__ == "__main__":
    main()
