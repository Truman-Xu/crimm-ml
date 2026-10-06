"""Equivariant basis matrices, with no `e3nn` and no `torch.jit.script`.

Adapted from `se3_transformer/model/basis.py`. Two changes, both required for
the model to be traceable and shippable:

1. `e3nn` is gone. `o3.wigner_3j` was already `lru_cache`d because it is a pure
   function of three integers -- it is a constant table, now baked into
   `_wigner_3j_data.py`. `o3.spherical_harmonics` is genuinely data-dependent
   and lives in `spherical_harmonics.py`, vendored verbatim.
2. `@torch.jit.script` dropped. A ScriptFunction is opaque to dynamo, so
   leaving it would graph-break in the middle of the basis computation, right
   where the edge geometry enters the model.

Dropped as unreachable for this configuration, not merely defaulted off:
`use_pad_trick`, `amp`, and the fully-fused basis. All three are gated on
`tensor_cores=True`, which `SE3TransformerWrapper` never sets -- the requested
fuse level is therefore always PARTIAL, and `ConvSE3` can never select its FULL
branch, the only consumer of `basis['fully_fused']`. See `layers.py`.
"""
from functools import lru_cache
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor

from ._wigner_3j_data import wigner_3j


def degree_to_dim(degree: int) -> int:
    return 2 * degree + 1


@lru_cache(maxsize=None)
def get_all_clebsch_gordon(max_degree: int, device, dtype=torch.float32) -> List[List[Tensor]]:
    """`[d_in * (max_degree+1) + d_out] -> [Q_J for J in |d_in-d_out| .. d_in+d_out]`.

    Cached per (max_degree, device): these never change, so this runs once per
    process and then costs a dict lookup.
    """
    all_cb = []
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            K_Js = []
            for J in range(abs(d_in - d_out), d_in + d_out + 1):
                K_Js.append(wigner_3j(J, d_in, d_out, device=device).to(dtype))
            all_cb.append(K_Js)
    return all_cb


def get_spherical_harmonics(relative_pos: Tensor, max_degree: int) -> List[Tensor]:
    """One tensor per degree 0 .. 2*max_degree, each `(E, 2l+1)`."""
    from .spherical_harmonics import spherical_harmonics
    lmax = 2 * max_degree
    sh = spherical_harmonics(lmax, relative_pos)
    return list(torch.split(sh, [degree_to_dim(d) for d in range(lmax + 1)], dim=1))


def get_basis(relative_pos: Tensor, max_degree: int = 4,
              compute_gradients: bool = False,
              clebsch_gordon: List[List[Tensor]] = None) -> Dict[str, Tensor]:
    """Pairwise bases `'{d_in},{d_out}' -> (E, 2*d_in+1, num_freq, 2*d_out+1)`.

    `clebsch_gordon` should be passed in by a caller that has already
    materialised the constants (`SE3Transformer` holds them as buffers); the
    `lru_cache`d fallback is for standalone use and for the verification script.
    """
    spherical_harmonics = get_spherical_harmonics(relative_pos, max_degree)
    if clebsch_gordon is None:
        clebsch_gordon = get_all_clebsch_gordon(max_degree, relative_pos.device)

    with torch.autograd.set_grad_enabled(compute_gradients):
        basis = {}
        idx = 0
        for d_in in range(max_degree + 1):
            for d_out in range(max_degree + 1):
                K_Js = []
                for freq_idx, J in enumerate(range(abs(d_in - d_out), d_in + d_out + 1)):
                    Q_J = clebsch_gordon[idx][freq_idx]
                    K_Js.append(torch.einsum(
                        "n f, k l f -> n l k", spherical_harmonics[J].float(), Q_J.float()))
                # stack on dim 2 so the layout is n l f k
                basis[f"{d_in},{d_out}"] = torch.stack(K_Js, 2)
                idx += 1
        return basis


def update_basis_with_fused(basis: Dict[str, Tensor], max_degree: int) -> Dict[str, Tensor]:
    """Add the per-output-degree and per-input-degree fused bases in place.

    `ConvSE3` picks one of these depending on which of its input or output
    channel counts is uniform; both are always built because that decision is
    made per layer, not per model.
    """
    # Built with pad + cat rather than by slice-assigning blocks into a
    # `torch.zeros` tensor (as upstream does). The values are bit-identical:
    # the blocks sit on the diagonal of the two block axes, so each block owns
    # a consecutive run of both, and zero padding supplies the rest. The
    # difference is in export: every slice assignment lowers to ScatterND
    # wrapped in edge-sized Transposes, ~1300 of them per forward. In ONNX
    # Runtime they cost ~250 ms of a ~1.6 s forward at L=76, and pad + cat
    # emits none.
    sum_dim = sum(degree_to_dim(d) for d in range(max_degree + 1))

    # Fused per output degree: blocks run down dim 1 (d_in), each padded to
    # its own slot on dim 2 (frequency).
    for d_out in range(max_degree + 1):
        sum_freq = sum(degree_to_dim(min(d, d_out)) for d in range(max_degree + 1))
        blocks, acc_f = [], 0
        for d_in in range(max_degree + 1):
            n_freq = degree_to_dim(min(d_out, d_in))
            block = basis[f"{d_in},{d_out}"][:, :, :, :degree_to_dim(d_out)]
            blocks.append(F.pad(block, (0, 0, acc_f, sum_freq - acc_f - n_freq)))
            acc_f += n_freq
        basis[f"out{d_out}_fused"] = torch.cat(blocks, dim=1)

    # Fused per input degree: blocks run down dim 2 (frequency), each padded
    # to its own slot on dim 3 (d_out).
    for d_in in range(max_degree + 1):
        blocks, acc_d = [], 0
        for d_out in range(max_degree + 1):
            dim_out = degree_to_dim(d_out)
            block = basis[f"{d_in},{d_out}"][:, :, :, :dim_out]
            blocks.append(F.pad(block, (acc_d, sum_dim - acc_d - dim_out)))
            acc_d += dim_out
        basis[f"in{d_in}_fused"] = torch.cat(blocks, dim=2)

    # The l = k = 0 basis is a constant; the pairwise conv special-cases its
    # absence rather than multiplying by it.
    del basis["0,0"]
    return basis
