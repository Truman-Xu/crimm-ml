"""Where the export and verification tools find things.

These tools are for building and checking the ONNX model, not for using it, so
they need two things this repo does not ship: the original PyTorch checkpoint
(`Base_ckpt.pt`) and upstream's IGSO(3) schedule pickles. Both live in an
RFdiffusion checkout. Point `RFDIFFUSION_DIR` at one; by default a checkout
next to this repo (`../RFdiffusion`) is used.

Importing this module also puts `src/` on `sys.path`, so the tools run from a
plain clone without `pip install -e .`.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # this repo
sys.path.insert(0, str(ROOT / "src"))

RF_REPO = Path(os.environ.get("RFDIFFUSION_DIR", ROOT.parent / "RFdiffusion"))
MODELS = RF_REPO / "models"
CKPT = MODELS / "Base_ckpt.pt"
UPSTREAM_SCHEDULES = RF_REPO / "schedules"

TEST_PDB = ROOT / "tests" / "data" / "2KL8.pdb"

