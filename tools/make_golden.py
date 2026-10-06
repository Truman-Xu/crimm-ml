"""Write the golden fixtures that `tests/test_golden.py` checks against.

    python tools/make_golden.py --model <rf_loop_dynamic_tp_ort.onnx> [--schedule-dir DIR]

Each fixture is one seeded design on `tests/data/2KL8.pdb`: the final
coordinates and pLDDT, plus `provenance.json` recording exactly what produced
them (package versions, model checksums, thread count). Regenerate only on
purpose -- after a change that is *meant* to alter the numbers, such as a new
model or a different random stream -- and commit the result with that change.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import crimm_ml  # noqa: E402
from crimm_ml import MinimalLoopSampler, load_model  # noqa: E402

PDB = ROOT / "tests" / "data" / "2KL8.pdb"
OUT = ROOT / "tests" / "fixtures"
CONTIG = "A1-20/6-6/A27-79"
CASES = [{"name": "golden_T15", "T": 15, "seed": 0}, {"name": "golden_T50", "T": 50, "seed": 0}]


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--schedule-dir", default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import onnxruntime as ort
    model_path = Path(args.model)
    model = load_model(model_path, device="cpu", threads=args.threads)
    pdb_lines = PDB.read_text().splitlines(keepends=True)
    files = [model_path, model_path.with_name(model_path.name + ".data")]
    checksums = {p.name: sha256(p) for p in files if p.exists()}

    for case in CASES:
        t0 = time.time()
        sampler = MinimalLoopSampler(pdb_lines, CONTIG, model, T=case["T"],
                                     schedule_dir=args.schedule_dir)
        design = sampler.sample(num_designs=1, seed=case["seed"], progress=False)[0]
        d = OUT / case["name"]
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "result.npz", x_final=design["x_final"], plddt_final=design["plddt_final"])
        (d / "provenance.json").write_text(json.dumps({
            "pdb": PDB.name, "contig": CONTIG, "T": case["T"], "seed": case["seed"],
            "model": checksums, "threads": args.threads, "device": "cpu",
            "crimm_ml": crimm_ml.__version__, "onnxruntime": ort.__version__,
            "numpy": np.__version__, "python": platform.python_version(),
            "machine": platform.machine(), "created": time.strftime("%Y-%m-%d"),
        }, indent=2) + "\n")
        print(f"{case['name']}: {time.time() - t0:.0f}s -> {d}")


if __name__ == "__main__":
    main()
