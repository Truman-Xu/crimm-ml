"""Phase 2: a from-scratch minimal reimplementation of RFdiffusion's
Sampler/SelfConditioning inference pipeline
(rfdiffusion/inference/model_runners.py), stripped to exactly what one use
case needs: a single motif+loop, one chain, no symmetry, no PPI hotspots, no
scaffold-guidance, no partial diffusion, no cyclic peptides.

The sampler is NumPy throughout and does not import torch. The network is a
separate object, loaded beforehand with `crimm_ml.models.load_model`
-- a PyTorch checkpoint or an ONNX export, called the same way -- so with an
ONNX model, sampling runs with no torch installed at all. The code around the
network is the NumPy port in `crimm_ml.numpy_rf`; its torch
originals are in `minimal_rf/` and serve as the reference.

Crimm-agnostic on purpose (plain PDB lines + a contig string in, arrays out)
-- `import crimm` never appears in this file. `crimm_ml.
crimm_utils.fill_gap_minimal` is the thin crimm-facing convenience wrapper.

Every kept code path is a direct transcription of the corresponding
model_runners.py logic (line numbers noted per method, against the
checked-out copy at rfdiffusion/inference/model_runners.py) -- nothing here
is a reinterpretation. What's NOT transcribed is simply absent, not
disabled-by-config:
  - Hydra `@hydra.main`/CLI config loading           (config built in Python)
  - checkpoint loading and model construction        (the caller passes a model)
  - SymGen / symmetry                                 (no import, no branches)
  - ScaffoldedSampler and everything under it          (not imported)
  - PPI/hotspot lookup (get_idx0_hotspots)             (hotspot_0idx=None)
  - cyclic-peptide branching (_init_cyclic_reses)       (the model builds its
                                                          own all-False mask)
  - partial-diffusion branch of sample_init             (full-generation only)
  - substrate/ligand potential setup                    (guiding_potentials
                                                          stays None; dead code)
  - run_inference.py's driver bookkeeping (design_startnum resume, cautious-
    skip, .trb pickling, trajectory writing, empty_cache_per_design)
"""
import os
import random
from importlib import resources

import numpy as np
from omegaconf import OmegaConf

from crimm_ml import assets
from crimm_ml.minimal_rf.contigs import ContigMap
from crimm_ml.minimal_rf.inference.pdb import parse_pdb_lines
from crimm_ml.minimal_rf.potentials.manager import PotentialManager
from crimm_ml.progress import make as progress_reporter
from crimm_ml.numpy_rf import geometry
from crimm_ml.numpy_rf.denoise import Denoise
from crimm_ml.numpy_rf.diffusion import Diffuser

F32 = np.float32


def _load_base_config(base_config_path, rfdiffusion_repo_dir):
    """`base.yaml`: the explicit path, else an RFdiffusion checkout's copy, else
    the copy packaged with crimm_ml (identical to upstream's)."""
    if base_config_path is not None:
        return OmegaConf.load(base_config_path)
    if rfdiffusion_repo_dir is not None:
        return OmegaConf.load(os.path.join(rfdiffusion_repo_dir, "config", "inference", "base.yaml"))
    text = resources.files("crimm_ml").joinpath("data").joinpath("base.yaml").read_text()
    return OmegaConf.create(text)


def _resolve_schedule_dir(schedule_dir, rfdiffusion_repo_dir):
    """The IGSO(3) table cache: the explicit directory, else an RFdiffusion
    checkout's `schedules/`, else crimm_ml's cache (`assets.schedule_dir()`).
    Nothing is created here; the cache is written only when a table is built."""
    if schedule_dir is not None:
        return str(schedule_dir)
    if rfdiffusion_repo_dir is not None:
        return os.path.join(rfdiffusion_repo_dir, "schedules")
    return str(assets.schedule_dir())


def assemble_config(model, T, base_config_path=None, rfdiffusion_repo_dir=None):
    """The inference config for `model` at `T` steps.

    model_runners.py:206-235 (assemble_config_from_chk, minus the
    Hydra-CLI-override-reapply loop at :237-250 -- irrelevant with no Hydra
    CLI), reading the checkpoint settings from the model. Our own overrides
    apply last, same as a CLI override would in the original (base.yaml's own
    T/contig values are placeholders anyway).
    """
    conf = _load_base_config(base_config_path, rfdiffusion_repo_dir)
    for cat in ("model", "diffuser", "preprocess"):
        for key in conf[cat]:
            if key in model.config[cat]:
                conf[cat][key] = model.config[cat][key]
    conf.diffuser.T = T
    return conf


def warm_schedules(model, Ts, schedule_dir=None, base_config_path=None,
                   rfdiffusion_repo_dir=None):
    """Build the IGSO(3) table for each T in `Ts` now, so that no sampler pays
    the ~1 minute on first use. Tables already cached are only read. Returns
    the cache directory."""
    schedule_dir = _resolve_schedule_dir(schedule_dir, rfdiffusion_repo_dir)
    for T in Ts:
        conf = assemble_config(model, T, base_config_path, rfdiffusion_repo_dir)
        Diffuser(**conf.diffuser, cache_dir=schedule_dir)
    return schedule_dir


def _process_target_from_lines(pdb_lines):
    """Inlined rfdiffusion/inference/utils.py:613-641 (`process_target`),
    minus the file-open at :615: takes PDB lines directly via
    `parse_pdb_lines`. `center=False` (matches how Sampler.initialize/
    sample_init always call process_target); `parse_hetatom` dropped (only
    used for ligand potentials, which this class doesn't support)."""
    parsed = parse_pdb_lines(pdb_lines, parse_hetatom=False)
    xyz = parsed["xyz"]
    seq_len = len(xyz)

    xyz_27 = np.full((seq_len, 27, 3), np.nan, dtype=F32)
    xyz_27[:, :14, :] = xyz[:, :14, :]

    mask_27 = np.zeros((seq_len, 27), dtype=bool)
    mask_27[:, :14] = parsed["mask"]

    return {
        "xyz_27": xyz_27,
        "mask_27": mask_27,
        "seq": parsed["seq"].astype(np.int64),
        "pdb_idx": parsed["pdb_idx"],
    }


class MinimalLoopSampler:
    """Minimal reimplementation of SelfConditioning for exactly one use
    case: fill a single missing loop into an otherwise-fixed motif.

    Usage:
        from crimm_ml.models import load_model

        model = load_model("models/rf_loop_dynamic_tp_ort.onnx")   # or Base_ckpt.pt
        sampler = MinimalLoopSampler(pdb_lines, contig, model, T=50)
        results = sampler.sample(num_designs=4, final_step=1)
        # results[i]["x_final"] -> (L,14,3) float32 backbone+placeholder coords

    Load the model once and pass it to as many samplers as you need; the
    sampler does not copy or modify it.
    """

    def __init__(
        self,
        pdb_lines,
        contig,
        model,
        T=50,
        base_config_path=None,
        schedule_dir=None,
        rfdiffusion_repo_dir=None,
    ):
        """`model` is an `RFModel` from `crimm_ml.models.load_model`.
        Its `config` supplies the checkpoint settings (diffusion schedule,
        preprocessing flags) that upstream reads from the checkpoint file.

        With an ONNX model nothing here imports torch; the pre- and
        post-processing is NumPy either way.

        Data files: `base_config_path` defaults to the `base.yaml` packaged
        with crimm_ml, and `schedule_dir` (the IGSO(3) table cache) to
        `crimm_ml.assets.schedule_dir()`. `rfdiffusion_repo_dir` points both
        at an RFdiffusion checkout instead, as before; explicit paths win.
        """
        schedule_dir = _resolve_schedule_dir(schedule_dir, rfdiffusion_repo_dir)

        self.model = model
        self.device = model.device

        conf = assemble_config(model, T, base_config_path, rfdiffusion_repo_dir)
        conf.contigmap.contigs = [contig]

        self.conf = conf
        self.diffuser_conf = conf.diffuser
        self.denoiser_conf = conf.denoiser
        self.preprocess_conf = conf.preprocess
        self.contig_conf = conf.contigmap
        self.potential_conf = conf.potentials
        self.T = int(conf.diffuser.T)

        # ---- :148 ----
        self.diffuser = Diffuser(**self.diffuser_conf, cache_dir=schedule_dir)

        # ---- :613-641, file-free (see _process_target_from_lines) ----
        self.target_feats = _process_target_from_lines(pdb_lines)

        # The translation noise upstream draws from torch; see numpy_rf/diffusion.py.
        self.noise_rng = np.random.default_rng()

    def _make_deterministic(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self.noise_rng = np.random.default_rng(seed)

    def sample(self, num_designs=1, final_step=1, deterministic=False, progress=None,
               seed=None):
        """Run the full reverse-diffusion loop for `num_designs` designs.
        Returns the same list[dict] shape upstream's sampler produces, with
        NumPy arrays in place of tensors, plus `"seed"`.

        `seed`: design `i` is seeded with `seed + i`, so a run can be repeated
        exactly. `deterministic=True` is the same as `seed=0`. With neither,
        the random state is left as the caller set it, and `"seed"` is None.
        Seeding sets the process-global `np.random` and `random` state (as
        upstream does), so concurrent samplers must run in separate processes.

        `progress` reports the reverse-diffusion steps (one bar over all
        designs). It defaults to a tqdm bar when tqdm is installed and the
        output is interactive, and takes `True`/`False`, a callback or your
        own bar -- see `crimm_ml.progress`.
        """
        if seed is None and deterministic:
            seed = 0
        results = []
        steps_per_design = self.T - final_step + 1   # `t_step_input` is set in _sample_init
        reporter = progress_reporter(progress, num_designs * steps_per_design)
        try:
            for i in range(num_designs):
                design_seed = None if seed is None else int(seed) + i
                if design_seed is not None:
                    self._make_deterministic(design_seed)

                x_init, seq_init = self._sample_init()
                x_t, seq_t = x_init.copy(), seq_init.copy()
                px0, plddt = None, None
                for t in range(self.t_step_input, final_step - 1, -1):
                    px0, x_t, seq_t, plddt = self._sample_step(
                        t=t, x_t=x_t, seq_init=seq_t, final_step=final_step
                    )
                    reporter.step({"design": i + 1, "num_designs": num_designs, "t": t})
                results.append(
                    {
                        "contig_map": self.contig_map,
                        "target_feats": self.target_feats,
                        "x_final": x_t,
                        "px0_final": px0,
                        "seq_init": seq_init,
                        "plddt_final": plddt,
                        "binderlen": self.binderlen,
                        "seed": design_seed,
                    }
                )
        finally:
            reporter.close()
        return results

    def _sample_init(self):
        """---- :292-475, full-generation branch only. The partial-diffusion
        branch (:411-424,441-445) and provide_seq handling (:458-459) are
        dropped entirely -- both are dead code when `diffuser.partial_T`
        never exists as a concept in this class. Ligand/substrate potential
        setup (:493-529) is dropped: only relevant when
        `potentials.guiding_potentials` mentions "substrate_contacts", which
        this class's config surface never sets. ----"""
        contig_map = ContigMap(self.target_feats, **self.contig_conf)
        self.contig_map = contig_map

        mask_seq = contig_map.inpaint_seq[None, :]
        mask_str = contig_map.inpaint_str[None, :]
        self.mask_seq = mask_seq
        self.mask_str = mask_str
        self.diffusion_mask = mask_str
        self.binderlen = len(contig_map.inpaint)

        # ---- :340-347 -- kept as a hard dependency of Denoise, a cheap no-op
        # here since potentials.guiding_potentials stays None ----
        potential_manager = PotentialManager(
            self.potential_conf,
            self.conf.ppi,
            self.diffuser_conf,
            self.conf.inference,
            None,  # hotspot_0idx -- get_idx0_hotspots dropped, always inert
            self.binderlen,
        )

        xyz_27 = self.target_feats["xyz_27"]
        mask_27 = self.target_feats["mask_27"]
        seq_orig = self.target_feats["seq"]
        L_mapped = len(contig_map.ref)

        # full-generation branch (:425-439)
        xyz_mapped = np.full((1, 1, L_mapped, 27, 3), np.nan, dtype=F32)
        xyz_mapped[:, :, contig_map.hal_idx0, ...] = xyz_27[contig_map.ref_idx0, ...]
        xyz_mapped = geometry.get_init_xyz(xyz_mapped).squeeze()
        atom_mask_mapped = np.zeros((L_mapped, 27), dtype=bool)
        atom_mask_mapped[contig_map.hal_idx0] = mask_27[contig_map.ref_idx0]

        self.t_step_input = self.T
        t_list = np.arange(1, self.t_step_input + 1)

        # sequence init (:452-462); provide_seq unmasking (:458-459) dropped
        seq_t = np.full((L_mapped,), 21, dtype=np.int64)
        seq_t[contig_map.hal_idx0] = seq_orig[contig_map.ref_idx0]
        seq_t[~mask_seq.squeeze()] = 21
        seq_t = np.eye(22, dtype=F32)[seq_t]

        fa_stack, _ = self.diffuser.diffuse_pose(
            xyz_mapped,
            seq_t.copy(),
            atom_mask_mapped.squeeze(),
            self.noise_rng,
            diffusion_mask=self.diffusion_mask.squeeze(),
            t_list=t_list,
        )
        xt = fa_stack[-1].squeeze()[:, :14, :].copy()

        # ---- :279-290 (construct_denoiser) ----
        denoise_kwargs = OmegaConf.to_container(self.diffuser_conf)
        denoise_kwargs.update(OmegaConf.to_container(self.denoiser_conf))
        denoise_kwargs.update(
            {
                "L": len(contig_map.ref),
                "diffuser": self.diffuser,
                "potential_manager": potential_manager,
            }
        )
        self.denoiser = Denoise(**denoise_kwargs)

        self.prev_pred = None
        return xt, seq_t

    def _preprocess(self, seq, xyz_t, t):
        """---- :566-719, minus the hotspot-feature branch at :683-707.
        That branch only ever runs `preprocess.d_t1d >= 24`; Base_ckpt.pt
        was trained without hotspot conditioning (d_t1d stays 22 after
        assemble_config_from_chk's checkpoint override), so it's dead code
        for this checkpoint, not something to special-case here. ----

        Returns the nine model inputs as NumPy arrays (float32; `idx` int64)."""
        L = seq.shape[0]

        msa_masked = np.zeros((1, 1, L, 48), dtype=F32)
        msa_masked[:, :, :, :22] = seq[None, None]
        msa_masked[:, :, :, 22:44] = seq[None, None]
        msa_masked[:, :, 0, 46] = 1.0
        msa_masked[:, :, -1, 47] = 1.0

        msa_full = np.zeros((1, 1, L, 25), dtype=F32)
        msa_full[:, :, :, :22] = seq[None, None]
        msa_full[:, :, 0, 23] = 1.0
        msa_full[:, :, -1, 24] = 1.0

        t1d = np.zeros((1, 1, L, 21), dtype=F32)
        seqt1d = seq.copy()
        masked = seqt1d[:, 21] == 1
        seqt1d[masked, 20] = 1
        seqt1d[masked, 21] = 0
        t1d[:, :, :, :21] = seqt1d[None, None, :, :21]

        mask_str = self.mask_str.squeeze()
        timefeature = np.zeros((L,), dtype=F32)
        timefeature[mask_str] = 1
        timefeature[~mask_str] = 1 - t / self.T
        timefeature = timefeature[None, None, ..., None]
        t1d = np.concatenate((t1d, timefeature), axis=-1)

        xyz_t = xyz_t.copy()
        xyz_t[~mask_str, 3:, :] = np.nan
        xyz_t = xyz_t[None, None]
        xyz_t = np.concatenate((xyz_t, np.full((1, 1, L, 13, 3), np.nan, dtype=F32)), axis=3)

        t2d = geometry.xyz_to_t2d(xyz_t)
        idx = np.array(self.contig_map.rf, dtype=np.int64)[None]

        seq_tmp = t1d[..., :-1].argmax(axis=-1).reshape(-1, L)
        alpha, _, alpha_mask, _ = geometry.get_torsions(
            xyz_t.reshape(-1, L, 27, 3), seq_tmp, geometry.torsion_indices,
            geometry.torsion_can_flip, geometry.reference_angles,
        )
        alpha_mask = np.logical_and(alpha_mask, ~np.isnan(alpha[..., 0]))
        alpha[np.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1, -1, L, 10, 2)
        alpha_mask = alpha_mask.reshape(1, -1, L, 10, 1)
        alpha_t = np.concatenate((alpha, alpha_mask.astype(F32)), axis=-1).reshape(1, -1, L, 30)

        return (
            msa_masked,
            msa_full,
            seq[None],
            xyz_t[0],
            idx,
            t1d,
            t2d,
            xyz_t,
            alpha_t,
        )

    def _sample_step(self, *, t, x_t, seq_init, final_step):
        """---- SelfConditioning.sample_step, :803-903, minus every
        `if self.symmetry is not None` branch (this class never supports
        symmetry). The self-conditioning guard simplifies from
        `(t < self.diffuser.T) and (t != self.diffuser_conf.partial_T)` to
        just `t < self.T`, since `partial_T` is not a concept this class
        has -- the comparison is always true for any real t. ----"""
        msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = (
            self._preprocess(seq_init, x_t, t)
        )
        B, N, L = xyz_t.shape[:3]

        if t < self.T:
            zeros = np.zeros((B, 1, L, 24, 3), dtype=F32)
            xyz_t = np.concatenate((self.prev_pred[:, None], zeros), axis=-2)
            t2d_44 = geometry.xyz_to_t2d(xyz_t)
        else:
            xyz_t = np.zeros_like(xyz_t)
            t2d_44 = np.zeros_like(t2d[..., :44])
        t2d[..., :44] = t2d_44

        # Dropped against the original call: msa_prev/pair_prev/state_prev
        # (always None here), `t` (unused by forward), return_infer (the only
        # path kept), motif_mask (freeze_track_motif is False for this
        # checkpoint, so the module builds its own all-False mask) and
        # cyclic_reses (all-False). See minimal_rf_module's docstring.
        _, _, px0, _, alpha, _, plddt = self.model(
            msa_masked, msa_full, seq_in, xt_in, idx_pdb,
            t1d, t2d, xyz_t, alpha_t,
        )

        self.prev_pred = px0.copy()

        _, px0 = geometry.compute_all_atom_coords(seq_in.argmax(axis=-1), px0, alpha)
        px0 = px0.squeeze(0)[:, :14]

        seq_t_1 = seq_init.copy()
        if t > final_step:
            x_t_1, px0 = self.denoiser.get_next_pose(
                xt=x_t,
                px0=px0,
                t=t,
                diffusion_mask=self.mask_str.squeeze(),
                noise_rng=self.noise_rng,
                align_motif=True,
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input,
            )
        else:
            x_t_1 = px0.copy()

        return px0, x_t_1, seq_t_1, plddt
