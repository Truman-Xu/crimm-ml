"""One reverse-diffusion step, without torch.

NumPy port of `rfdiffusion/inference/utils.py`: `get_next_frames`,
`get_mu_xt_x0`, `get_next_ca`, `get_noise_schedule` and `Denoise` (with an
empty potential manager, the only kind this pipeline builds). Coordinate
arrays are float32 as upstream; see `diffusion.py` for how the random draws
relate to upstream's.
"""
import logging

import numpy as np
from scipy.spatial.transform import Rotation as scipy_R

from .diffusion import get_beta_schedule, normal_like
from .geometry import rigid_from_3_points

F32 = np.float32


def get_next_frames(xt, px0, t, diffuser, so3_type, diffusion_mask, noise_scale=1.0):
    """Backbone coordinates at t-1 from the IGSO(3) reverse step. [L, 3, 3] float64."""
    R_0, _ = rigid_from_3_points(px0[None, :, 0, :], px0[None, :, 1, :], px0[None, :, 2, :])
    R_t, Ca_t = rigid_from_3_points(xt[None, :, 0, :], xt[None, :, 1, :], xt[None, :, 2, :])
    R_0 = scipy_R.from_matrix(R_0.squeeze()).as_matrix()
    R_t = scipy_R.from_matrix(R_t.squeeze()).as_matrix()

    L = R_t.shape[0]
    all_rot_transitions = np.broadcast_to(np.identity(3), (L, 3, 3)).copy()
    if so3_type != "igso3":
        raise NotImplementedError(f"so3 diffusion type {so3_type} not implemented")
    all_rot_transitions[~diffusion_mask] = diffuser.so3_diffuser.reverse_sample_vectorized(
        R_t[~diffusion_mask], R_0[~diffusion_mask], t,
        noise_level=noise_scale, mask=None, return_perturb=True,
    )
    all_rot_transitions = all_rot_transitions[:, None, :, :]
    Ca_t = Ca_t.squeeze()
    next_crds = (np.einsum("lrij,laj->lrai", all_rot_transitions, xt[:, :3, :] - Ca_t[:, None, ...])
                 + Ca_t[:, None, None, ...])
    return next_crds.squeeze(1)


def get_mu_xt_x0(xt, px0, t, beta_schedule, alphabar_schedule, eps=1e-6):
    t_idx = t - 1
    sigma = ((1 - alphabar_schedule[t_idx - 1]) / (1 - alphabar_schedule[t_idx])) * beta_schedule[t_idx]
    xt_ca = xt[:, 1, :]
    px0_ca = px0[:, 1, :]
    a = ((np.sqrt(alphabar_schedule[t_idx - 1] + eps) * beta_schedule[t_idx])
         / (1 - alphabar_schedule[t_idx])) * px0_ca
    b = ((np.sqrt(1 - beta_schedule[t_idx] + eps) * (1 - alphabar_schedule[t_idx - 1]))
         / (1 - alphabar_schedule[t_idx])) * xt_ca
    return a + b, sigma


def get_next_ca(xt, px0, t, diffusion_mask, crd_scale, beta_schedule, alphabar_schedule,
                noise_rng, noise_scale=1.0):
    px0 = px0 * crd_scale
    xt = xt * crd_scale
    mu, sigma = get_mu_xt_x0(xt, px0, t, beta_schedule=beta_schedule, alphabar_schedule=alphabar_schedule)
    sampled_crds = normal_like(noise_rng, mu, np.sqrt(sigma * noise_scale))
    delta = sampled_crds - xt[:, 1, :]
    if diffusion_mask is not None:
        delta[diffusion_mask, ...] = 0
    out_crds = xt + delta[:, None, :]
    return out_crds / crd_scale, delta / crd_scale


def get_noise_schedule(T, noiseT, noise1, schedule_type):
    noise_schedules = {
        "constant": lambda t: noiseT,
        "linear": lambda t: ((t - 1) / (T - 1)) * (noiseT - noise1) + noise1,
    }
    assert schedule_type in noise_schedules, (
        f"noise_schedule must be one of {noise_schedules.keys()}. Received {schedule_type}."
    )
    return noise_schedules[schedule_type]


class Denoise:
    """x(t-1) from x(t) and the network's prediction of x(0): CA by
    re-diffusing towards the prediction, frames by the IGSO(3) score."""

    def __init__(self, T, L, diffuser, b_0=0.001, b_T=0.1, min_b=1.0, max_b=12.5, min_sigma=0.05,
                 max_sigma=1.5, noise_level=0.5, schedule_type="linear", so3_schedule_type="linear",
                 schedule_kwargs={}, so3_type="igso3", noise_scale_ca=1.0, final_noise_scale_ca=1,
                 ca_noise_schedule_type="constant", noise_scale_frame=0.5, final_noise_scale_frame=0.5,
                 frame_noise_schedule_type="constant", crd_scale=1 / 15, potential_manager=None,
                 partial_T=None):
        self.T = T
        self.L = L
        self.diffuser = diffuser
        self.b_0 = b_0
        self.b_T = b_T
        self.noise_level = noise_level
        self.schedule_type = schedule_type
        self.so3_type = so3_type
        self.crd_scale = crd_scale
        self.potential_manager = potential_manager
        self._log = logging.getLogger(__name__)
        self.schedule, self.alpha_schedule, self.alphabar_schedule = get_beta_schedule(
            self.T, self.b_0, self.b_T, self.schedule_type, inference=True)
        self.noise_schedule_ca = get_noise_schedule(self.T, noise_scale_ca, final_noise_scale_ca,
                                                    ca_noise_schedule_type)
        self.noise_schedule_frame = get_noise_schedule(self.T, noise_scale_frame, final_noise_scale_frame,
                                                       frame_noise_schedule_type)
        if potential_manager is not None and not potential_manager.is_empty():
            raise NotImplementedError("guiding potentials are not supported")

    def align_to_xt_motif(self, px0, xT, diffusion_mask, eps=1e-6):
        """Rigidly move px0 so its motif sits on xT's motif (Kabsch)."""
        def rmsd(V, W, eps=0):
            N = V.shape[-2]
            return np.sqrt(np.sum((V - W) * (V - W), axis=(-2, -1)) / N + eps)

        assert xT.shape[1] == px0.shape[1], f"xT has shape {xT.shape} and px0 has shape {px0.shape}"
        L, n_atom, _ = xT.shape
        atom_mask = ~np.isnan(px0)
        px0 = px0.copy()

        px0_motif = px0[diffusion_mask, :3].reshape(-1, 3)
        xT_motif = xT[diffusion_mask, :3].reshape(-1, 3)
        px0_motif_mean = np.copy(px0_motif.mean(0))
        xT_motif_mean = np.copy(xT_motif.mean(0))
        A = xT_motif - xT_motif_mean
        B = px0_motif - px0_motif_mean

        U, S, Vt = np.linalg.svd(np.matmul(A.T, B))
        d = np.eye(3)
        d[-1, -1] = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ d @ U.T
        self._log.debug("Sampled motif RMSD: %.2f", rmsd(A, B @ R))   # once per step

        px0[~atom_mask] = 0
        px0 = px0.reshape(-1, 3) - px0_motif_mean
        px0_ = px0 @ R + xT_motif_mean
        px0_ = px0_.reshape([L, n_atom, 3])
        px0_[~atom_mask] = float("nan")
        return px0_.astype(F32)

    def get_next_pose(self, xt, px0, t, diffusion_mask, noise_rng, fix_motif=True, align_motif=True,
                      include_motif_sidechains=True):
        """xt, px0 [L, 14, 3] float32; diffusion_mask [L] bool (True = fixed)."""
        assert xt.shape[1] in (14, 27) and px0.shape[1] in (14, 27)
        if align_motif and diffusion_mask.any():
            px0 = self.align_to_xt_motif(px0, xt, diffusion_mask)
        if not fix_motif:
            diffusion_mask = np.zeros_like(diffusion_mask)

        _, ca_deltas = get_next_ca(
            xt, px0, t, diffusion_mask, crd_scale=self.crd_scale, beta_schedule=self.schedule,
            alphabar_schedule=self.alphabar_schedule, noise_rng=noise_rng,
            noise_scale=self.noise_schedule_ca(t),
        )
        frames_next = get_next_frames(
            xt, px0, t, diffuser=self.diffuser, so3_type=self.so3_type,
            diffusion_mask=diffusion_mask, noise_scale=self.noise_schedule_frame(t),
        )
        # Upstream adds `guide_scale(t) * potential_gradient` here; with no
        # potentials configured that gradient is all zeros.
        frames_next = frames_next + ca_deltas[:, None, :]

        fullatom_next = np.full((1,) + xt.shape, np.nan, dtype=F32)
        fullatom_next[:, :, :3] = frames_next[None]
        if include_motif_sidechains:
            fullatom_next[:, diffusion_mask, :14] = xt[None, diffusion_mask]
        return fullatom_next.squeeze(0)[:, :14, :], px0
