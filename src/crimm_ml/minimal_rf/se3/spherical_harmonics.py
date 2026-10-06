"""Real spherical harmonics up to degree 4, vendored from `e3nn` 0.3.3.

The SE(3)-Transformer needs `e3nn` for exactly two things, and both are removed
here so nothing outside torch is imported at runtime:

  * `o3.spherical_harmonics(...)` -- this file. Data-dependent (it is a function
    of each edge's direction), so it has to stay a real computation. It is
    already pure torch arithmetic, so vendoring it is a copy, not a rewrite.
  * `o3.wigner_3j(...)` -- constants. Baked into `basis.py` as literals.

Copied verbatim from `e3nn/o3/_spherical_harmonics.py` (the closed-form
polynomial recursion `_spherical_harmonics`, truncated at lmax=4) plus the
`normalize` / `normalization='integral'` handling from `SphericalHarmonics.
forward`, which is how `basis.py` calls it. Verbatim on purpose: these
polynomials are a fixed convention, and the model's weights were trained
against this one, so any "tidier" reformulation is a silent behaviour change.
Equivalence against the installed e3nn is asserted in
`examples/verify_no_dgl.py`.

lmax=4 is generous headroom. The bases need degrees up to `2 * max_degree`, and
`Base_ckpt.pt` sets `num_degrees=2` -> `max_degree=1` -> degrees 0,1,2 only.

The `@torch.jit.script` decorator e3nn puts on this is deliberately dropped:
a ScriptFunction is opaque to dynamo, which would put a graph break in the
middle of the basis computation. As plain Python it traces straight through.
"""
import math

import torch
from torch import Tensor

MAX_LMAX = 4


def _spherical_harmonics(lmax: int, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """e3nn's polynomial recursion. Returns `(..., (lmax+1)**2)`, degrees
    concatenated in order 0, 1, ..., lmax."""
    sh_0_0 = torch.ones_like(x)
    if lmax == 0:
        return torch.stack([sh_0_0], dim=-1)

    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    if lmax == 1:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2], dim=-1)

    sh_2_0 = math.sqrt(3.0) * x * z
    sh_2_1 = math.sqrt(3.0) * x * y
    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)
    sh_2_2 = y2 - 0.5 * x2z2
    sh_2_3 = math.sqrt(3.0) * y * z
    sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))
    if lmax == 2:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        ], dim=-1)

    sh_3_0 = math.sqrt(5.0 / 6.0) * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = math.sqrt(5.0) * sh_2_0 * y
    sh_3_2 = math.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x
    sh_3_3 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = math.sqrt(3.0 / 8.0) * z * (4.0 * y2 - x2z2)
    sh_3_5 = math.sqrt(5.0) * sh_2_4 * y
    sh_3_6 = math.sqrt(5.0 / 6.0) * (sh_2_4 * z - sh_2_0 * x)
    if lmax == 3:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        ], dim=-1)

    sh_4_0 = 0.935414346693485 * sh_3_0 * z + 0.935414346693485 * sh_3_6 * x
    sh_4_1 = 0.661437827766148 * sh_3_0 * y + 0.810092587300982 * sh_3_1 * z + 0.810092587300983 * sh_3_5 * x
    sh_4_2 = (-0.176776695296637 * sh_3_0 * z + 0.866025403784439 * sh_3_1 * y
              + 0.684653196881458 * sh_3_2 * z + 0.684653196881457 * sh_3_4 * x
              + 0.176776695296637 * sh_3_6 * x)
    sh_4_3 = (-0.306186217847897 * sh_3_1 * z + 0.968245836551855 * sh_3_2 * y
              + 0.790569415042095 * sh_3_3 * x + 0.306186217847897 * sh_3_5 * x)
    sh_4_4 = -0.612372435695795 * sh_3_2 * x + sh_3_3 * y - 0.612372435695795 * sh_3_4 * z
    sh_4_5 = (-0.306186217847897 * sh_3_1 * x + 0.790569415042096 * sh_3_3 * z
              + 0.968245836551854 * sh_3_4 * y - 0.306186217847897 * sh_3_5 * z)
    sh_4_6 = (-0.176776695296637 * sh_3_0 * x - 0.684653196881457 * sh_3_2 * x
              + 0.684653196881457 * sh_3_4 * z + 0.866025403784439 * sh_3_5 * y
              - 0.176776695296637 * sh_3_6 * z)
    sh_4_7 = -0.810092587300982 * sh_3_1 * x + 0.810092587300982 * sh_3_5 * z + 0.661437827766148 * sh_3_6 * y
    sh_4_8 = -0.935414346693485 * sh_3_0 * x + 0.935414346693486 * sh_3_6 * z
    if lmax == 4:
        return torch.stack([
            sh_0_0,
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
            sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        ], dim=-1)

    raise ValueError(f"lmax={lmax} exceeds the vendored maximum of {MAX_LMAX}")


def spherical_harmonics(lmax: int, vectors: Tensor) -> Tensor:
    """`e3nn.o3.spherical_harmonics(range(lmax+1), vectors, normalize=True)`
    with the default `normalization='integral'`.

    Returns `(..., (lmax+1)**2)`. `F.normalize` (not a plain division) so a
    zero-length edge yields zeros rather than NaN -- e3nn's own choice, and
    load-bearing here because a self-edge would otherwise poison the basis.
    """
    if lmax > MAX_LMAX:
        raise ValueError(f"lmax={lmax} exceeds the vendored maximum of {MAX_LMAX}")
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    sh = _spherical_harmonics(lmax, vectors[..., 0], vectors[..., 1], vectors[..., 2])

    # normalization='integral': scale degree l by sqrt(2l+1)/sqrt(4*pi).
    # Built as a constant vector so this is one multiply, matching e3nn's
    # `sh.mul_(torch.cat([...]))`.
    scale = torch.cat([
        (math.sqrt(2 * l + 1) / math.sqrt(4 * math.pi))
        * torch.ones(2 * l + 1, dtype=sh.dtype, device=sh.device)
        for l in range(lmax + 1)
    ])
    return sh * scale
