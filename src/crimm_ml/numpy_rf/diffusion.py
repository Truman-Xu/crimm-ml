"""Forward diffusion of backbone frames, without torch.

NumPy port of `rfdiffusion/diffusion.py` (`get_beta_schedule`,
`EuclideanDiffuser`, `IGSO3`, `Diffuser.diffuse_pose`), limited to the
linear schedules and IGSO(3) frames the sampler uses.

Random numbers
--------------
Upstream draws from two generators: rotations from NumPy's global
`np.random` (IGSO(3) angles and axes, reverse-step rotation noise), and
translations from torch (`torch.normal`). The NumPy draws are kept exactly as
they were, so for a given `np.random.seed` they are the same numbers as
upstream. The torch draws cannot be reproduced without torch; they come from
a separate `numpy.random.Generator` (`noise_rng`) as float32 standard
normals, applied as `z * std + mean` -- the arithmetic `torch.normal(mean,
std)` performs. Trajectories are therefore statistically identical to
upstream but not the same numbers for the same seed; with the torch draws
replaced by the same generator (as the equivalence tests do), they agree to
float rounding.
"""
import contextlib
import logging
import os
import pickle
import uuid

import numpy as np
from scipy.spatial.transform import Rotation as scipy_R

from . import igso3
from .geometry import rigid_from_3_points, torch_linspace_f32

F32 = np.float32

log = logging.getLogger(__name__)


def _write_npz_atomic(path, arrays):
    """Write `path` so that readers only ever see a complete file.

    The temporary name is unique per writer, so concurrent processes building
    the same table never write into one file; `os.replace` then makes the last
    writer win atomically, and every writer produces the same arrays. Caching
    is best effort: a read-only or full cache directory costs a recompute next
    time, not a failed run.
    """
    tmp = f"{path[:-len('.npz')]}.tmp.{os.getpid()}.{uuid.uuid4().hex}.npz"
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    except OSError as e:
        log.warning("Could not cache IGSO(3) tables at %s (%s); they will be "
                    "recomputed next time", path, e)
        with contextlib.suppress(OSError):
            os.remove(tmp)


def get_beta_schedule(T, b0, bT, schedule_type, schedule_params={}, inference=False):
    assert schedule_type in ["linear"]
    assert T >= 15, "With discrete time and T < 15, the schedule is badly approximated"
    b0 *= 200 / T
    bT *= 200 / T
    schedule = torch_linspace_f32(b0, bT, T)
    alpha_schedule = 1 - schedule
    alphabar_t_schedule = np.cumprod(alpha_schedule, dtype=F32)
    if inference:
        log.debug("With this beta schedule (%s schedule, beta_0 = %s, beta_T = %s), "
                  "alpha_bar_T = %s", schedule_type, round(b0, 3), round(bT, 3),
                  alphabar_t_schedule[-1])
    return schedule, alpha_schedule, alphabar_t_schedule


def normal_like(noise_rng, mean, std):
    """`torch.normal(mean, std)`: standard normals scaled then shifted."""
    shape = np.broadcast_shapes(np.shape(mean), np.shape(std))
    z = noise_rng.standard_normal(shape, dtype=F32)
    return z * std + mean


class EuclideanDiffuser:
    def __init__(self, T, b_0, b_T, schedule_type="linear", schedule_kwargs={}):
        self.T = T
        self.beta_schedule, self.alpha_schedule, self.alphabar_schedule = get_beta_schedule(
            T, b_0, b_T, schedule_type, **schedule_kwargs
        )

    def diffuse_translations(self, xyz, noise_rng, diffusion_mask=None, var_scale=1):
        return self.apply_kernel_recursive(xyz, noise_rng, diffusion_mask, var_scale)

    def apply_kernel(self, x, t, noise_rng, diffusion_mask=None, var_scale=1):
        assert len(x.shape) == 3
        L = x.shape[0]
        ca_xyz = x[:, 1, :]
        b_t = self.beta_schedule[t - 1]
        mean = np.sqrt(1 - b_t) * ca_xyz
        var = np.ones((L, 3), dtype=F32) * b_t * var_scale
        sampled_crds = normal_like(noise_rng, mean, np.sqrt(var))
        delta = sampled_crds - ca_xyz
        if diffusion_mask is not None:
            delta[diffusion_mask, ...] = 0
        return x + delta[:, None, :], delta

    def apply_kernel_recursive(self, xyz, noise_rng, diffusion_mask=None, var_scale=1):
        bb_stack, T_stack = [], []
        cur_xyz = xyz.copy()
        for t in range(1, self.T + 1):
            cur_xyz, cur_T = self.apply_kernel(cur_xyz, t, noise_rng, var_scale=var_scale,
                                               diffusion_mask=diffusion_mask)
            bb_stack.append(cur_xyz)
            T_stack.append(cur_T)
        return np.stack(bb_stack).swapaxes(0, 1), np.stack(T_stack).swapaxes(0, 1)


def _torch_scalar(t):
    """How `torch.tensor(t)` types a scalar: a Python number becomes float32,
    a NumPy scalar keeps its dtype. The IGSO(3) schedule inherits that."""
    if isinstance(t, (np.generic, np.ndarray)):   # before `float`: np.float64 subclasses it
        return np.asarray(t)
    return F32(t)


class IGSO3:
    def __init__(self, *, T, min_sigma, max_sigma, min_b, max_b, cache_dir,
                 num_omega=1000, schedule="linear", L=2000):
        if schedule != "linear":
            raise NotImplementedError("only the linear SO(3) schedule is ported")
        self.T = T
        self.schedule = schedule
        self.cache_dir = cache_dir
        self.min_sigma = min_sigma
        self.min_b = min_b
        self.max_b = max_b
        self.max_sigma = self.sigma(1.0)
        self.num_omega = num_omega
        self.num_sigma = 500
        self.L = L
        self.igso3_vals = self._calc_igso3_vals(L=L)
        self.step_size = 1 / self.T

    def _calc_igso3_vals(self, L=2000):
        """IGSO(3) tables, cached in `cache_dir`.

        Upstream caches them as a pickle (`<name>.pkl`), and in the caches it
        wrote, `discrete_sigma` is a torch tensor, so unpickling needs torch.
        This port keeps its own cache as plain arrays (`<name>.npz`): it
        converts an existing pickle when that loads, and otherwise computes
        the tables with `igso3.calculate_igso3` (a one-off cost).
        """
        rp = lambda x: str(x).replace(".", "_")
        stem = os.path.join(
            self.cache_dir,
            f"T_{self.T}_omega_{self.num_omega}_min_sigma_{rp(self.min_sigma)}"
            f"_min_b_{rp(self.min_b)}_max_b_{rp(self.max_b)}_schedule_{self.schedule}",
        )
        if os.path.exists(stem + ".npz"):
            with np.load(stem + ".npz", allow_pickle=False) as z:
                return {k: z[k] for k in z.files}

        vals = None
        if os.path.exists(stem + ".pkl"):
            try:
                with open(stem + ".pkl", "rb") as fh:
                    vals = pickle.load(fh)
            except ImportError:   # the pickle holds torch tensors and torch is absent
                vals = None
        if vals is None:
            log.info("Computing IGSO(3) tables for T=%d (about a minute; cached in %s)",
                     self.T, self.cache_dir)
            vals = igso3.calculate_igso3(num_sigma=self.num_sigma, min_sigma=self.min_sigma,
                                         max_sigma=self.max_sigma, num_omega=self.num_omega)
        vals = {k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v)) for k, v in vals.items()}
        _write_npz_atomic(stem + ".npz", vals)
        return vals

    @property
    def discrete_sigma(self):
        return self.igso3_vals["discrete_sigma"]

    def sigma_idx(self, sigma):
        return np.digitize(sigma, self.discrete_sigma) - 1

    def t_to_idx(self, t):
        return self.sigma_idx(self.sigma(t / self.T))

    def sigma(self, t):
        t = _torch_scalar(t)
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f"Invalid t={t}")
        return self.min_sigma + t * self.min_b + (1 / 2) * (t ** 2) * (self.max_b - self.min_b)

    def g(self, t):
        """sqrt(d sigma(t)^2 / dt), evaluated the way torch autograd does in
        float32 (the torch code differentiates `sigma(t) ** 2`)."""
        t = F32(t)
        two_sigma = F32(2) * self.sigma(t)
        grad = two_sigma * F32(self.min_b) + ((two_sigma * F32(self.max_b - self.min_b)) * F32(0.5)) * (F32(2) * t)
        return np.sqrt(grad)

    def sample(self, ts, n_samples=1):
        assert sum(ts == 0) == 0, "assumes one-indexed, not zero indexed"
        all_samples = []
        for t in ts:
            sigma_idx = self.t_to_idx(t)
            all_samples.append(np.interp(np.random.rand(n_samples),
                                         self.igso3_vals["cdf"][sigma_idx],
                                         self.igso3_vals["discrete_omega"]))
        return np.stack(all_samples, axis=0)

    def sample_vec(self, ts, n_samples=1):
        x = np.random.randn(len(ts), n_samples, 3)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return x * self.sample(ts, n_samples=n_samples)[..., None]

    def score_norm(self, t, omega):
        sigma_idx = self.t_to_idx(t)
        return np.interp(omega, self.igso3_vals["discrete_omega"], self.igso3_vals["score_norm"][sigma_idx])

    def diffuse_frames(self, xyz, t_list, diffusion_mask=None):
        """xyz [L, 3, 3] float32 -> perturbed coordinates [L, T, 3, 3] (float64)."""
        t = np.arange(self.T) + 1
        num_res = len(xyz)
        R_true, Ca = rigid_from_3_points(xyz[None, :, 0, :], xyz[None, :, 1, :], xyz[None, :, 2, :])
        R_true, Ca = R_true[0], Ca[0]

        sampled_rots = self.sample_vec(t, n_samples=num_res)
        if diffusion_mask is not None:
            sampled_rots = sampled_rots * (1 - diffusion_mask[None, :, None])

        R_sampled = scipy_R.from_rotvec(sampled_rots.reshape(-1, 3)).as_matrix().reshape(self.T, num_res, 3, 3)
        R_perturbed = np.einsum("tnij,njk->tnik", R_sampled, R_true)
        perturbed_crds = (np.einsum("tnij,naj->tnai", R_sampled, xyz[:, :3, :] - Ca[:, None, ...])
                          + Ca[None, :, None])
        if t_list is not None:
            idx = [i - 1 for i in t_list]
            perturbed_crds = perturbed_crds[idx]
            R_perturbed = R_perturbed[idx]
        return perturbed_crds.transpose(1, 0, 2, 3), R_perturbed.transpose(1, 0, 2, 3)

    def reverse_sample_vectorized(self, R_t, R_0, t, noise_level, mask=None, return_perturb=False):
        """R_t, R_0 [N, 3, 3] float64; t a Python int. See the torch original
        for the derivation (de Bortoli et al. 2022; Song et al. 2020)."""
        R_0t = R_t @ np.swapaxes(R_0, -1, -2)
        R_0t_rotvec = scipy_R.from_matrix(R_0t).as_rotvec()
        Omega = np.linalg.norm(R_0t_rotvec, axis=-1)
        Score_approx = R_0t_rotvec * (self.score_norm(t, Omega) / Omega)[:, None]

        rot_g = self.g(t / self.T)                       # float32, as in torch
        Z = np.random.normal(size=(R_0.shape[0], 3))
        Z *= noise_level
        Delta_r = ((rot_g ** 2) * F32(self.step_size)) * Score_approx
        Perturb_tangent = Delta_r + (rot_g * F32(np.sqrt(self.step_size))) * Z
        if mask is not None:
            Perturb_tangent *= (1 - mask.astype(np.int64))[:, None, None]
        Perturb = igso3.Exp(Perturb_tangent)
        if return_perturb:
            return Perturb
        return Perturb @ R_t


class Diffuser:
    def __init__(self, T, b_0, b_T, min_sigma, max_sigma, min_b, max_b, schedule_type,
                 so3_schedule_type, so3_type, crd_scale, schedule_kwargs={}, var_scale=1.0,
                 cache_dir=".", partial_T=None, truncation_level=2000):
        self.T = T
        self.b_0 = b_0
        self.b_T = b_T
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.crd_scale = crd_scale
        self.var_scale = var_scale
        self.cache_dir = cache_dir
        self.so3_diffuser = IGSO3(T=self.T, min_sigma=self.min_sigma, max_sigma=self.max_sigma,
                                  schedule=so3_schedule_type, min_b=min_b, max_b=max_b,
                                  cache_dir=self.cache_dir, L=truncation_level)
        self.eucl_diffuser = EuclideanDiffuser(self.T, b_0, b_T, schedule_type=schedule_type, **schedule_kwargs)
        log.debug("Successful diffuser __init__")

    def diffuse_pose(self, xyz, seq, atom_mask, noise_rng, include_motif_sidechains=True,
                     diffusion_mask=None, t_list=None):
        """xyz [L, 27, 3] float32; diffusion_mask [L] bool, True = not diffused."""
        if diffusion_mask is None:
            diffusion_mask = np.zeros(len(xyz.squeeze()), dtype=bool)
        assert not np.isnan(xyz.squeeze()[:, :3]).any(), "backbone atoms must be present"

        if np.sum(diffusion_mask) != 0:
            self.motif_com = xyz[diffusion_mask, 1, :].mean(axis=0)
            xyz = xyz - self.motif_com
        else:
            xyz = xyz - xyz[:, 1, :].mean(axis=0)

        xyz_true = xyz.copy()
        xyz = xyz * self.crd_scale

        diffused_T, deltas = self.eucl_diffuser.diffuse_translations(
            xyz[:, :3, :].copy(), noise_rng, diffusion_mask=diffusion_mask)
        diffused_T /= self.crd_scale
        deltas /= self.crd_scale

        diffused_frame_crds, _ = self.so3_diffuser.diffuse_frames(
            xyz[:, :3, :].copy(), diffusion_mask=diffusion_mask, t_list=None)
        diffused_frame_crds /= self.crd_scale

        cum_delta = np.cumsum(deltas, axis=1, dtype=F32)
        diffused_BB = (diffused_frame_crds + cum_delta[:, :, None, :]).swapaxes(0, 1)
        t_steps, L = diffused_BB.shape[:2]

        diffused_fa = np.zeros((t_steps, L, 27, 3), dtype=F32)
        diffused_fa[:, :, :3, :] = diffused_BB
        if include_motif_sidechains:
            diffused_fa[:, diffusion_mask, :14, :] = xyz_true[None, diffusion_mask, :14]

        if t_list is None:
            fa_stack = diffused_fa
        else:
            fa_stack = diffused_fa[[t - 1 for t in t_list]]
        return fa_stack, xyz_true
