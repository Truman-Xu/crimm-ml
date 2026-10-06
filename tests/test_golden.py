"""Seeded designs must reproduce the committed fixtures.

The fixtures (`tools/make_golden.py`) pin the numbers the sampler produces for
a fixed model, seed and input. A change that is not meant to alter results --
a refactor, a dependency bump -- must keep these passing; one that is (a new
model, a different random stream) regenerates them in the same commit.

The tolerance, 1e-3 Å, is far above the ~1e-5 Å that ONNX Runtime versions,
thread counts and CPUs move the coordinates, and far below the ångström-scale
change any real difference in the sampler produces.
"""
import hashlib
import json

import numpy as np
import pytest

from conftest import FIXTURES

TOL = 1e-3


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


@pytest.mark.parametrize("name", ["golden_T15", pytest.param("golden_T50", marks=pytest.mark.slow)])
def test_golden(name, onnx_model_path, pdb_lines):
    from crimm_ml import MinimalLoopSampler, load_model

    prov = json.loads((FIXTURES / name / "provenance.json").read_text())
    want = prov["model"].get(onnx_model_path.name)
    if want is not None and _sha256(onnx_model_path) != want:
        pytest.fail(f"{onnx_model_path.name} is not the model {name} was made with; if the "
                    f"model changed on purpose, regenerate with tools/make_golden.py")

    model = load_model(onnx_model_path, device="cpu", threads=prov["threads"])
    sampler = MinimalLoopSampler(pdb_lines, prov["contig"], model, T=prov["T"])
    design = sampler.sample(num_designs=1, seed=prov["seed"], progress=False)[0]

    ref = np.load(FIXTURES / name / "result.npz")
    assert design["seed"] == prov["seed"]
    dx = float(np.max(np.abs(design["x_final"] - ref["x_final"])))
    dp = float(np.max(np.abs(design["plddt_final"] - ref["plddt_final"])))
    assert dx < TOL, f"x_final moved by {dx:.2e} Å (tolerance {TOL})"
    assert dp < TOL, f"pLDDT moved by {dp:.2e} (tolerance {TOL})"
