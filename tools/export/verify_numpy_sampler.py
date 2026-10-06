"""Checks the NumPy sampler against the torch pipeline it was ported from.

`MinimalLoopSampler` runs NumPy around the network (`crimm_ml.
numpy_rf`). This compares it with `TorchReferenceSampler` below, which is the
sampler as it was before the port: the same logic on the torch originals in
`minimal_rf/` (`diffusion.py`, `inference/utils.py`, `kinematics.py`,
`util.py`, `util_module.ComputeAllAtomCoords`).

Randomness is the one thing that cannot match by itself: upstream draws the
translation noise from torch, the port from NumPy. For the comparison the
reference's `torch.normal` is replaced by draws from the same NumPy generator,
in the same order, so both runs see identical noise; everything else
(`np.random`, `random`) is seeded identically anyway.

Phases (all by default; pass names to pick):
  tables   residue tables built in NumPy vs torch
  igso3    IGSO(3) tables computed in NumPy vs the torch-written cache
           (only needed when no cache exists; see numpy_rf/diffusion.py)
  funcs    frames, torsions, all-atom coordinates, template features, init xyz
  sampler  full trajectories, NumPy sampler vs torch reference, same model
  onnx     NumPy sampler with an ONNX model vs with the PyTorch model
  notorch  the ONNX sampler in a subprocess where `import torch` is blocked

Usage:
    python tools/export/verify_numpy_sampler.py [phases] \\
        [--onnx models/rf_loop_dynamic_tp_ort.onnx] [--T 50] [--threads 8]
"""
import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from _paths import CKPT, MODELS, ROOT, TEST_PDB, UPSTREAM_SCHEDULES  # noqa: E402  (also puts src/ on sys.path)

import numpy as np

PDB = TEST_PDB
CONTIG = "A1-20/6/A30-79"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def maxabs(a, b):
    a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
    b = b.detach().cpu().numpy() if hasattr(b, "detach") else np.asarray(b)
    both = ~(np.isnan(a) | np.isnan(b))
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    return float(np.abs(a[both].astype(np.float64) - b[both]).max()) if both.any() else 0.0


def check(name, diff, tol):
    ok = diff <= tol
    log(f"  {name:44s} max|diff| {diff:.2e}  (tol {tol:.0e})  {'OK' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------ reference
class TorchReferenceSampler:
    """`MinimalLoopSampler` as it was before the NumPy port (2026-09-17), with
    the model passed in rather than loaded. Torch tensors throughout."""

    def __init__(self, pdb_lines, contig, module, config, T=50, device="cpu"):
        import torch
        from omegaconf import OmegaConf
        from crimm_ml.minimal_rf.diffusion import Diffuser
        from crimm_ml.minimal_rf.util_module import ComputeAllAtomCoords
        from crimm_ml.minimal_rf.inference.utils import parse_pdb_lines

        self.device = torch.device(device)
        conf = OmegaConf.load(ROOT / "src" / "crimm_ml" / "data" / "base.yaml")
        for cat in ("model", "diffuser", "preprocess"):
            for key in conf[cat]:
                if key in config[cat]:
                    conf[cat][key] = config[cat][key]
        conf.diffuser.T = T
        conf.contigmap.contigs = [contig]
        self.conf = conf
        self.diffuser_conf = conf.diffuser
        self.denoiser_conf = conf.denoiser
        self.preprocess_conf = conf.preprocess
        self.contig_conf = conf.contigmap
        self.potential_conf = conf.potentials
        self.T = int(conf.diffuser.T)
        self.model = module
        self.diffuser = Diffuser(**self.diffuser_conf, cache_dir=str(UPSTREAM_SCHEDULES))
        self.allatom = ComputeAllAtomCoords().to(self.device)

        parsed = parse_pdb_lines(pdb_lines, parse_hetatom=False)
        xyz = torch.from_numpy(parsed["xyz"])
        L = len(xyz)
        xyz_27 = torch.full((L, 27, 3), float("nan")).float()
        xyz_27[:, :14, :] = xyz[:, :14, :]
        mask_27 = torch.full((L, 27), False)
        mask_27[:, :14] = torch.from_numpy(parsed["mask"])
        self.target_feats = {"xyz_27": xyz_27, "mask_27": mask_27,
                             "seq": torch.from_numpy(parsed["seq"]), "pdb_idx": parsed["pdb_idx"]}

    def sample(self, num_designs=1, final_step=1, deterministic=False):
        import torch
        results = []
        for i in range(num_designs):
            if deterministic:
                torch.manual_seed(i)
                np.random.seed(i)
                random.seed(i)
            x_init, seq_init = self._sample_init()
            x_t, seq_t = x_init.clone(), seq_init.clone()
            px0 = plddt = None
            for t in range(self.t_step_input, final_step - 1, -1):
                px0, x_t, seq_t, plddt = self._sample_step(t=t, x_t=x_t, seq_init=seq_t,
                                                           final_step=final_step)
            results.append({"x_final": x_t, "px0_final": px0, "plddt_final": plddt,
                            "seq_init": seq_init})
        return results

    def _sample_init(self):
        import torch
        import torch.nn.functional as nn
        from omegaconf import OmegaConf
        from crimm_ml.minimal_rf.contigs import ContigMap
        from crimm_ml.minimal_rf.inference.utils import Denoise
        from crimm_ml.minimal_rf.kinematics import get_init_xyz
        from crimm_ml.minimal_rf.potentials.manager import PotentialManager

        contig_map = ContigMap(self.target_feats, **self.contig_conf)
        self.contig_map = contig_map
        self.mask_seq = torch.from_numpy(contig_map.inpaint_seq)[None, :]
        self.mask_str = torch.from_numpy(contig_map.inpaint_str)[None, :]
        self.diffusion_mask = self.mask_str
        self.binderlen = len(contig_map.inpaint)
        pm = PotentialManager(self.potential_conf, self.conf.ppi, self.diffuser_conf,
                              self.conf.inference, None, self.binderlen)
        xyz_27, mask_27, seq_orig = (self.target_feats[k] for k in ("xyz_27", "mask_27", "seq"))
        L_mapped = len(contig_map.ref)
        xyz_mapped = torch.full((1, 1, L_mapped, 27, 3), float("nan"))
        xyz_mapped[:, :, contig_map.hal_idx0, ...] = xyz_27[contig_map.ref_idx0, ...]
        xyz_mapped = get_init_xyz(xyz_mapped).squeeze()
        atom_mask_mapped = torch.full((L_mapped, 27), False)
        atom_mask_mapped[contig_map.hal_idx0] = mask_27[contig_map.ref_idx0]
        self.t_step_input = self.T
        t_list = np.arange(1, self.t_step_input + 1)
        seq_t = torch.full((1, L_mapped), 21).squeeze()
        seq_t[contig_map.hal_idx0] = seq_orig[contig_map.ref_idx0]
        seq_t[~self.mask_seq.squeeze()] = 21
        seq_t = nn.one_hot(seq_t, num_classes=22).float()
        fa_stack, _ = self.diffuser.diffuse_pose(
            xyz_mapped, torch.clone(seq_t), atom_mask_mapped.squeeze(),
            diffusion_mask=self.diffusion_mask.squeeze(), t_list=t_list)
        xt = torch.clone(fa_stack[-1].squeeze()[:, :14, :])
        kw = OmegaConf.to_container(self.diffuser_conf)
        kw.update(OmegaConf.to_container(self.denoiser_conf))
        kw.update({"L": L_mapped, "diffuser": self.diffuser, "potential_manager": pm})
        self.denoiser = Denoise(**kw)
        self.prev_pred = None
        return xt, seq_t

    def _preprocess(self, seq, xyz_t, t):
        import torch
        from crimm_ml.minimal_rf import util
        from crimm_ml.minimal_rf.kinematics import xyz_to_t2d
        L = seq.shape[0]
        msa_masked = torch.zeros((1, 1, L, 48))
        msa_masked[:, :, :, :22] = seq[None, None]
        msa_masked[:, :, :, 22:44] = seq[None, None]
        msa_masked[:, :, 0, 46] = 1.0
        msa_masked[:, :, -1, 47] = 1.0
        msa_full = torch.zeros((1, 1, L, 25))
        msa_full[:, :, :, :22] = seq[None, None]
        msa_full[:, :, 0, 23] = 1.0
        msa_full[:, :, -1, 24] = 1.0
        t1d = torch.zeros((1, 1, L, 21))
        seqt1d = torch.clone(seq)
        for idx in range(L):
            if seqt1d[idx, 21] == 1:
                seqt1d[idx, 20] = 1
                seqt1d[idx, 21] = 0
        t1d[:, :, :, :21] = seqt1d[None, None, :, :21]
        tf = torch.zeros((L)).float()
        tf[self.mask_str.squeeze()] = 1
        tf[~self.mask_str.squeeze()] = 1 - t / self.T
        t1d = torch.cat((t1d, tf[None, None, ..., None]), dim=-1).float()
        xyz_t = xyz_t.clone()
        xyz_t[~self.mask_str.squeeze(), 3:, :] = float("nan")
        xyz_t = torch.cat((xyz_t[None, None], torch.full((1, 1, L, 13, 3), float("nan"))), dim=3)
        t2d = xyz_to_t2d(xyz_t)
        idx = torch.tensor(self.contig_map.rf)[None]
        seq_tmp = t1d[..., :-1].argmax(dim=-1).reshape(-1, L)
        alpha, _, alpha_mask, _ = util.get_torsions(
            xyz_t.reshape(-1, L, 27, 3), seq_tmp, util.torsion_indices,
            util.torsion_can_flip, util.reference_angles)
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[..., 0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha_t = torch.cat((alpha.reshape(1, -1, L, 10, 2), alpha_mask.reshape(1, -1, L, 10, 1)),
                            dim=-1).reshape(1, -1, L, 30)
        d = self.device
        return (msa_masked.to(d), msa_full.to(d), seq[None].to(d), torch.squeeze(xyz_t, dim=0).to(d),
                idx.to(d), t1d.to(d), t2d.to(d), xyz_t.to(d), alpha_t.to(d))

    def _sample_step(self, *, t, x_t, seq_init, final_step):
        import torch
        from crimm_ml.minimal_rf.kinematics import xyz_to_t2d
        msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = \
            self._preprocess(seq_init, x_t, t)
        B, N, L = xyz_t.shape[:3]
        if t < self.T:
            zeros = torch.zeros(B, 1, L, 24, 3).float().to(xyz_t.device)
            xyz_t = torch.cat((self.prev_pred.unsqueeze(1), zeros), dim=-2)
            t2d_44 = xyz_to_t2d(xyz_t)
        else:
            xyz_t = torch.zeros_like(xyz_t)
            t2d_44 = torch.zeros_like(t2d[..., :44])
        t2d[..., :44] = t2d_44
        with torch.no_grad():
            _, _, px0, _, alpha, _, plddt = self.model(
                msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t)
        self.prev_pred = torch.clone(px0)
        _, px0 = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0 = px0.squeeze()[:, :14]
        if t > final_step:
            x_t_1, px0 = self.denoiser.get_next_pose(
                xt=x_t, px0=px0, t=t, diffusion_mask=self.mask_str.squeeze(), align_motif=True,
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input)
        else:
            x_t_1 = torch.clone(px0).to(x_t.device)
            px0 = px0.to(x_t.device)
        return px0, x_t_1, torch.clone(seq_init), plddt


class _SharedNoise:
    """Replaces `torch.normal(mean, std)` with the NumPy sampler's draws:
    float32 standard normals from a `default_rng(seed)`, applied as
    `z * std + mean` (torch's own order)."""

    def __init__(self):
        self.rng = None

    def __enter__(self):
        import torch
        self._orig = torch.normal

        def normal(mean, std, *a, **k):
            shape = np.broadcast_shapes(tuple(mean.shape), tuple(std.shape))
            z = torch.from_numpy(self.rng.standard_normal(shape, dtype=np.float32)).to(mean.device)
            return z * std + mean

        torch.normal = normal
        return self

    def __exit__(self, *exc):
        import torch
        torch.normal = self._orig


# ---------------------------------------------------------------- phases
def phase_tables(args):
    import torch
    from crimm_ml.minimal_rf import util as TU
    from crimm_ml.numpy_rf import geometry as G
    log("tables: NumPy vs torch")
    ok = True
    # Exact, except that the chi frames in RTs_by_torsion may differ by one
    # float32 step (1.2e-7): torch's vector-norm kernel rounds differently from
    # every NumPy formulation tried (sqrt-of-sum in any order, float64, BLAS
    # nrm2, pow), so `make_frame` cannot reproduce it bit for bit.
    for name, tol in (("torsion_indices", 0.0), ("torsion_can_flip", 0.0), ("base_indices", 0.0),
                      ("xyzs_in_base_frame", 0.0), ("RTs_by_torsion", 2e-7), ("reference_angles", 0.0)):
        a, b = getattr(G, name), getattr(TU, name).numpy()
        same = a.dtype == b.dtype and a.shape == b.shape
        ok &= check(f"{name} {a.dtype}{a.shape}",
                    maxabs(a.astype(np.float64), b.astype(np.float64)) if same else float("inf"), tol)
    return ok


def phase_igso3(args):
    import pickle
    import torch  # noqa: F401  (the upstream cache holds a torch tensor)
    from crimm_ml.numpy_rf import igso3 as NI
    name = "T_50_omega_1000_min_sigma_0_02_min_b_1_5_max_b_2_5_schedule_linear.pkl"
    log(f"igso3: NumPy-computed tables vs the torch-written cache {name}")
    with open(UPSTREAM_SCHEDULES / name, "rb") as fh:
        ref = pickle.load(fh)
    t0 = time.time()
    from types import SimpleNamespace
    from crimm_ml.numpy_rf.diffusion import IGSO3
    # the linear schedule replaces the configured max_sigma with sigma(1.0), as upstream does
    max_sigma = IGSO3.sigma(SimpleNamespace(min_sigma=0.02, min_b=1.5, max_b=2.5), 1.0)
    got = NI.calculate_igso3(num_sigma=500, num_omega=1000, min_sigma=0.02, max_sigma=max_sigma)
    log(f"  computed in {time.time() - t0:.0f}s")
    ref = {k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v)) for k, v in ref.items()}

    def rel(a, b, mask=None):
        mask = np.ones(a.shape, bool) if mask is None else mask
        return float(np.max(np.abs(a[mask] - b[mask]) / np.maximum(np.abs(b[mask]), 1e-12)))

    ok = True
    for k in ("discrete_omega", "discrete_sigma", "cdf"):
        ok &= check(f"{k} (relative)", rel(got[k], ref[k]), 1e-9)
    # exp_score_norms (unused by sampling) inherits NaN from rows whose score
    # has underflowed; compare the rows that are finite in both.
    fin = np.isfinite(ref["exp_score_norms"]) & np.isfinite(got["exp_score_norms"])
    ok &= check(f"exp_score_norms, finite rows ({fin.sum()}/{fin.size})",
                rel(got["exp_score_norms"], ref["exp_score_norms"], fin), 1e-9)
    # The score is round-off wherever the density has underflowed (large
    # angles, small sigma), in upstream's table and in ours alike, so it is
    # only compared where the density is meaningful.
    pdf = np.diff(np.concatenate([np.zeros((ref["cdf"].shape[0], 1)), ref["cdf"]], axis=1)) \
        * ref["cdf"].shape[1] / np.pi
    meaningful = (pdf > 1e-9) & np.isfinite(ref["score_norm"])
    ok &= check(f"score_norm where density > 1e-9 ({meaningful.mean() * 100:.0f}%)",
                rel(got["score_norm"], ref["score_norm"], meaningful), 1e-5)
    return ok


def phase_funcs(args):
    import torch
    from crimm_ml.minimal_rf import util as TU
    from crimm_ml.minimal_rf.kinematics import get_init_xyz, xyz_to_t2d
    from crimm_ml.minimal_rf.util_module import ComputeAllAtomCoords
    from crimm_ml.numpy_rf import geometry as G
    log("functions: NumPy vs torch on real coordinates")
    rng = np.random.default_rng(0)
    from crimm_ml.minimal_rf.inference.pdb import parse_pdb_lines
    p = parse_pdb_lines(open(PDB).readlines())
    L = len(p["xyz"])
    xyz27 = np.full((1, L, 27, 3), np.nan, np.float32)
    xyz27[0, :, :14] = np.where(p["mask"][..., None], p["xyz"], np.nan)
    seq = p["seq"][None].astype(np.int64)
    ok = True

    R, T = G.rigid_from_3_points(xyz27[..., 0, :], xyz27[..., 1, :], xyz27[..., 2, :])
    Rt, Tt = TU.rigid_from_3_points(*(torch.from_numpy(xyz27[..., k, :]) for k in range(3)))
    ok &= check("rigid_from_3_points", maxabs(R, Rt), 1e-6)

    a = G.get_torsions(xyz27, seq, G.torsion_indices, G.torsion_can_flip, G.reference_angles)
    b = TU.get_torsions(torch.from_numpy(xyz27), torch.from_numpy(seq), TU.torsion_indices,
                        TU.torsion_can_flip, TU.reference_angles)
    for n, x, y in zip(("torsions", "torsions_alt", "tors_mask", "tors_planar"), a, b):
        ok &= check(f"get_torsions: {n}", maxabs(x.astype(np.float64), y.double()), 1e-5)

    alphas = rng.standard_normal((1, L, 10, 2)).astype(np.float32)
    _, xa = G.compute_all_atom_coords(seq, xyz27[:, :, :3], alphas)
    _, xb = ComputeAllAtomCoords()(torch.from_numpy(seq), torch.from_numpy(xyz27[:, :, :3]),
                                   torch.from_numpy(alphas))
    ok &= check("compute_all_atom_coords", maxabs(xa, xb), 1e-4)

    xyz_t = xyz27[None].copy()
    xyz_t[..., 3:, :] = np.nan
    for Ls in (L, 20):   # >25 points (matrix-product cdist) and <=25 (exact)
        ta = G.xyz_to_t2d(xyz_t[:, :, :Ls])
        tb = xyz_to_t2d(torch.from_numpy(xyz_t[:, :, :Ls]))
        ok &= check(f"xyz_to_t2d L={Ls}", maxabs(ta, tb), 1e-5)
        ok &= check(f"xyz_to_t2d L={Ls} distance bins identical", float(np.abs(ta[..., :37] - tb[..., :37].numpy()).max()), 0.0)

    gap = xyz27[None].copy()
    gap[:, :, 20:26] = np.nan
    ok &= check("get_init_xyz", maxabs(G.get_init_xyz(gap), get_init_xyz(torch.from_numpy(gap))), 1e-5)
    return ok


def _run_numpy_sampler(model, T, seed=0):
    from crimm_ml.minimal_loop_sampler import MinimalLoopSampler
    s = MinimalLoopSampler(open(PDB).readlines(), CONTIG, model, T=T)
    t0 = time.time()
    out = s.sample(num_designs=1, deterministic=True)[0]
    return out, time.time() - t0


def phase_sampler(args):
    import torch
    from crimm_ml.models.torch_model import TorchRFModel
    log(f"sampler: NumPy sampler vs torch reference, T={args.T}, identical noise")
    rf = TorchRFModel(CKPT, device="cpu")
    ref = TorchReferenceSampler(open(PDB).readlines(), CONTIG, rf.module, rf.config, T=args.T)
    with _SharedNoise() as noise:
        noise.rng = np.random.default_rng(0)   # the NumPy sampler reseeds its own the same way
        t0 = time.time()
        r = ref.sample(num_designs=1, deterministic=True)[0]
        t_ref = time.time() - t0
    n, t_np = _run_numpy_sampler(rf, args.T)
    log(f"  torch reference {t_ref:.1f}s, NumPy sampler {t_np:.1f}s")
    ok = check("x_final (A)", maxabs(n["x_final"], r["x_final"]), 1e-3)
    ok &= check("px0_final (A)", maxabs(n["px0_final"], r["px0_final"]), 1e-3)
    ok &= check("plddt_final", maxabs(n["plddt_final"], r["plddt_final"]), 1e-3)
    return ok


def phase_onnx(args):
    from crimm_ml.models import load_model
    log(f"onnx: NumPy sampler with {Path(args.onnx).name} vs with the PyTorch model, T={args.T}")
    a, ta = _run_numpy_sampler(load_model(CKPT, device="cpu"), args.T)
    b, tb = _run_numpy_sampler(load_model(args.onnx, device="cpu", threads=args.threads), args.T)
    log(f"  PyTorch model {ta:.1f}s, ONNX model {tb:.1f}s")
    ok = check("x_final (A)", maxabs(a["x_final"], b["x_final"]), 1e-3)
    ok &= check("plddt_final", maxabs(a["plddt_final"], b["plddt_final"]), 1e-3)
    return ok


NO_TORCH = r'''
import importlib.abc, sys, time
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"torch is blocked in this test (import of {{name!r}})")
sys.meta_path.insert(0, Block())
sys.path.insert(0, {repo!r})
from crimm_ml.models import load_model
from crimm_ml.minimal_loop_sampler import MinimalLoopSampler
try:  # the crimm-facing wrapper must not need torch either (crimm itself may be absent)
    from crimm_ml import crimm_utils  # noqa: F401
except ImportError as e:
    if "torch" in str(e):
        raise
model = load_model({onnx!r}, device={device!r}, threads={threads})
s = MinimalLoopSampler(open({pdb!r}).readlines(), {contig!r}, model, T={T})
t0 = time.time()
out = s.sample(num_designs=1, deterministic=True)[0]
x = out["x_final"]
print("NOTORCH device", model.device, "providers", model.providers[0],
      f"sample {{time.time() - t0:.1f}}s", "x_final range", float(x.min()), float(x.max()),
      "torch in sys.modules:", "torch" in sys.modules)
'''


def phase_notorch(args):
    log(f"notorch: ONNX sampler in a subprocess with torch imports blocked (device={args.device})")
    code = NO_TORCH.format(repo=str(ROOT / "src"), onnx=str(args.onnx), device=args.device,
                           threads=args.threads, pdb=str(PDB), contig=CONTIG, T=args.T)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith("NOTORCH")]
    if p.returncode != 0 or not lines:
        log("  FAILED:\n" + (p.stderr[-2000:] or p.stdout[-2000:]))
        return False
    log("  " + lines[-1])
    return "torch in sys.modules: False" in lines[-1]


PHASES = {"tables": phase_tables, "igso3": phase_igso3, "funcs": phase_funcs, "sampler": phase_sampler,
          "onnx": phase_onnx, "notorch": phase_notorch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phases", nargs="*", default=list(PHASES))
    ap.add_argument("--onnx", default=str(MODELS / "rf_loop_dynamic_tp_ort.onnx"))
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="device for the notorch phase")
    args = ap.parse_args()
    if any(p in args.phases for p in ("tables", "igso3", "funcs", "sampler", "onnx")):
        import torch
        torch.set_num_threads(args.threads)
        log(f"torch {torch.__version__}, numpy {np.__version__}")
    results = {}
    for p in args.phases:
        results[p] = PHASES[p](args)
    log("SUMMARY: " + ", ".join(f"{p}={'PASS' if ok else 'FAIL'}" for p, ok in results.items()))
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
