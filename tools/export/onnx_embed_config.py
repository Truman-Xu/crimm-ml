"""Store the checkpoint's settings inside existing ONNX exports.

`crimm_ml.models.load_model` reads the diffusion, preprocessing and
architecture settings of an ONNX model from its metadata, so the model can be
used with no checkpoint file and no torch. `export_onnx.py` writes them;
this adds them to files exported before it did. Needs torch and `onnx`, once,
on the machine that has the checkpoint. Only the graph file is rewritten; the
`.data` weights file is untouched.

Usage:
    python tools/export/onnx_embed_config.py models/rf_loop_dynamic_tp_ort.onnx [...]
    python tools/export/onnx_embed_config.py <model.onnx> --checkpoint models/Base_ckpt.pt
"""
import argparse
import sys
from pathlib import Path

from _paths import CKPT  # noqa: E402  (also puts src/ on sys.path)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--checkpoint", default=str(CKPT))
    args = ap.parse_args()

    import torch
    from crimm_ml.models.base import CONFIG_SECTIONS
    from crimm_ml.models.onnx_model import embed_config

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = {k: dict(ckpt["config_dict"][k]) for k in CONFIG_SECTIONS}
    for path in args.models:
        embed_config(path, config)
        print(f"embedded the config of {Path(args.checkpoint).name} into {path}")


if __name__ == "__main__":
    main()
