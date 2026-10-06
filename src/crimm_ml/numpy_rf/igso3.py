"""IGSO(3) tables and the SO(3) exponential map, without torch.

NumPy port of `rfdiffusion/igso3.py` (`calculate_igso3`, `Exp`). The torch
original differentiates log f(omega) with autograd; here the derivative is
written out, which is the same expression autograd evaluates. Everything is
float64, as it is upstream when the tables are built: `max_sigma` arrives there
as a torch tensor (`IGSO3.sigma(1.0)`), which makes `discrete_sigma` a float64
tensor and every `t = sigma**2` a float64 tensor too.

The sampler normally reads these tables from the cache in `schedules/` (see
`diffusion.IGSO3._calc_igso3_vals`); this module only runs when none exists.
"""
import numpy as np
from scipy.spatial.transform import Rotation

L_default = 2000


def Exp(A):
    """Rotation vectors [..., 3] -> rotation matrices [..., 3, 3]."""
    return Rotation.from_rotvec(A).as_matrix()


def _weights(t, L):
    ls = np.arange(L, dtype=np.int64)
    t = np.float64(t)
    return (2 * ls + 1) * np.exp(-ls * (ls + 1) * t / 2)


def f_igso3(omega, t, L=L_default):
    """Truncated IGSO(3) series (Leach et al. 2022, eq. 5)."""
    w = _weights(t, L)
    k = np.arange(L, dtype=np.float64) + 0.5
    omega = np.asarray(omega, dtype=np.float64)[:, None]
    return (w * np.sin(omega * k) / np.sin(omega / 2)).sum(axis=-1)


def d_logf_d_omega(omega, t, L=L_default):
    w = _weights(t, L)
    k = np.arange(L, dtype=np.float64) + 0.5
    omega = np.asarray(omega, dtype=np.float64)[:, None]
    s = np.sin(omega / 2)
    num = np.sin(omega * k)
    f = (w * num / s).sum(axis=-1)
    df = (w * (np.cos(omega * k) * k / s - num * np.cos(omega / 2) * 0.5 / s ** 2)).sum(axis=-1)
    return df / f


def igso3_density_angle(omega, t, L=L_default):
    return f_igso3(omega, t, L) * (1 - np.cos(omega)) / np.pi


def calculate_igso3(*, num_sigma, num_omega, min_sigma, max_sigma):
    """Same tables as `rfdiffusion.igso3.calculate_igso3`.

    Where the truncated density underflows (large angles at small sigma, below
    ~1e-9) the score is round-off, and it differs from upstream's -- including
    where it is inf or NaN -- because no two implementations cancel the same
    way there. Everywhere else it agrees to ~1e-8 relative, and a full
    sampling run with these tables gives the same coordinates as with
    upstream's (`examples/verify_numpy_sampler.py igso3`).
    """
    discrete_omega = np.linspace(0, np.pi, num_omega + 1)[1:]
    discrete_sigma = 10 ** np.linspace(np.log10(min_sigma), np.log10(max_sigma), num_sigma + 1)[1:]
    with np.errstate(all="ignore"):   # upstream computes the underflow region silently too
        pdf_vals = np.asarray([igso3_density_angle(discrete_omega, sigma ** 2) for sigma in discrete_sigma])
        cdf_vals = np.asarray([pdf.cumsum() / num_omega * np.pi for pdf in pdf_vals])
        score_norm = np.asarray([d_logf_d_omega(discrete_omega, sigma ** 2) for sigma in discrete_sigma])
        exp_score_norms = np.sqrt(np.sum(score_norm ** 2 * pdf_vals, axis=1) / np.sum(pdf_vals, axis=1))
    return {
        "cdf": cdf_vals,
        "score_norm": score_norm,
        "exp_score_norms": exp_score_norms,
        "discrete_omega": discrete_omega,
        "discrete_sigma": discrete_sigma,
    }
