"""Branch-free reimplementations of the `rfdiffusion.Embeddings`/
`rfdiffusion.Track_module` classes that carry a `use_checkpoint` branch:
`TemplatePairStack`, `Templ_emb`, `IterBlock`, `IterativeSimulator`. Used by
`minimal_rf_module.MinimalRoseTTAFoldModule` in place of the originals.

`use_checkpoint` is gradient checkpointing -- a training-time compute/memory
tradeoff (recompute a block's activations during backward instead of
storing them, to fit a larger model in GPU memory). It only matters when
there's a backward pass to save memory for. `minimal_rf_module.py`'s module
docstring already established `use_checkpoint` is never `True` anywhere in
this codebase for this pipeline (`SelfConditioning.sample_step` runs
entirely under `torch.no_grad()`). `MinimalRoseTTAFoldModule` itself only
went as far as calling the *originals* with `use_checkpoint=False`; this
module goes one level deeper and actually removes the dead
`if use_checkpoint: ... else: ...` branches from `TemplatePairStack.forward`,
`Templ_emb.forward`, and `IterBlock.forward`, rather than just choosing the
right one at the call site.

Everything else about these classes -- `__init__`, submodule types,
parameter names/shapes -- is unchanged from the originals, so a
checkpoint's state dict loads into these classes exactly the way it loads
into `rfdiffusion.Embeddings.Templ_emb`/`rfdiffusion.Track_module.
IterativeSimulator`: same attribute names, same shapes, same keys.
Wherever a submodule has no `use_checkpoint` branch of its own
(`MSAPairStr2MSA`, `MSA2Pair`, `PairStr2Pair`, `Str2Str`, and every
attention/feedforward/dropout building block in `Attention_module.py`/
`util_module.py` -- confirmed by grepping both files for `use_checkpoint`/
`checkpoint.`: no matches), it's imported and reused directly, unmodified.

Note: `TemplateTorsionStack` (`Embeddings.py:168-197`) also has a
`use_checkpoint` branch, but it's dead code independent of this pipeline --
`Templ_emb.__init__` never instantiates it (`self.tor_stack = ...` is
commented out in the original source, for every config). Not reproduced
here: there is no code path, for any config, that reaches it, so there is
nothing to strip.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# `torch.einsum`, not `opt_einsum.contract` -- matching the patch applied
# throughout `crimm_ml/minimal_rf/`. dynamo cannot trace
# opt_einsum's pure-Python `contract`, and mixing the two here would make this
# module disagree with the rest of the package at the 1e-3 level. See
# `minimal_rf/__init__.py` for the rationale and measurements.
from torch import einsum

from crimm_ml.minimal_rf.Attention_module import Attention
from crimm_ml.minimal_rf.Track_module import MSA2Pair, MSAPairStr2MSA, PairStr2Pair, Str2Str
from crimm_ml.minimal_rf.util_module import init_lecun_normal, pairwise_distance, rbf


class MinimalTemplatePairStack(nn.Module):
    """`TemplatePairStack` (`Embeddings.py:148-166`) with the
    `use_checkpoint` branch removed."""

    def __init__(self, n_block=2, d_templ=64, n_head=4, d_hidden=16, p_drop=0.25):
        super().__init__()
        self.n_block = n_block
        self.block = nn.ModuleList(
            [PairStr2Pair(d_pair=d_templ, n_head=n_head, d_hidden=d_hidden, p_drop=p_drop) for _ in range(n_block)]
        )
        self.norm = nn.LayerNorm(d_templ)

    def forward(self, templ, rbf_feat):
        B, T, L = templ.shape[:3]
        templ = templ.reshape(B * T, L, L, -1)
        for i_block in range(self.n_block):
            templ = self.block[i_block](templ, rbf_feat)
        return self.norm(templ).reshape(B, T, L, L, -1)


class MinimalTemplEmb(nn.Module):
    """`Templ_emb` (`Embeddings.py:199-288`) with both `use_checkpoint`
    branches removed, using `MinimalTemplatePairStack` for its own
    `templ_stack` submodule (in place of `TemplatePairStack`)."""

    def __init__(self, d_t1d=21 + 1 + 1, d_t2d=43 + 1, d_tor=30, d_pair=128, d_state=32,
                 n_block=2, d_templ=64, n_head=4, d_hidden=16, p_drop=0.25):
        super().__init__()
        self.emb = nn.Linear(d_t1d * 2 + d_t2d, d_templ)
        self.templ_stack = MinimalTemplatePairStack(
            n_block=n_block, d_templ=d_templ, n_head=n_head, d_hidden=d_hidden, p_drop=p_drop,
        )
        self.attn = Attention(d_pair, d_templ, n_head, d_hidden, d_pair)
        self.emb_t1d = nn.Linear(d_t1d + d_tor, d_templ)
        self.proj_t1d = nn.Linear(d_templ, d_templ)
        self.attn_tor = Attention(d_state, d_templ, n_head, d_hidden, d_state)

        self.emb = init_lecun_normal(self.emb)
        nn.init.zeros_(self.emb.bias)
        nn.init.kaiming_normal_(self.emb_t1d.weight, nonlinearity="relu")
        nn.init.zeros_(self.emb_t1d.bias)
        self.proj_t1d = init_lecun_normal(self.proj_t1d)
        nn.init.zeros_(self.proj_t1d.bias)

    def forward(self, t1d, t2d, alpha_t, xyz_t, pair, state):
        B, T, L, _ = t1d.shape

        left = t1d.unsqueeze(3).expand(-1, -1, -1, L, -1)
        right = t1d.unsqueeze(2).expand(-1, -1, L, -1, -1)
        templ = torch.cat((t2d, left, right), -1)
        templ = self.emb(templ)
        xyz_t = xyz_t.reshape(B * T, L, -1, 3)
        rbf_feat = rbf(pairwise_distance(xyz_t[:, :, 1]))
        templ = self.templ_stack(templ, rbf_feat)

        t1d = torch.cat((t1d, alpha_t), dim=-1)
        t1d = self.proj_t1d(F.relu_(self.emb_t1d(t1d)))

        state = state.reshape(B * L, 1, -1)
        t1d = t1d.permute(0, 2, 1, 3).reshape(B * L, T, -1)
        out = self.attn_tor(state, t1d, t1d).reshape(B, L, -1)
        state = state.reshape(B, L, -1)
        state = state + out

        pair = pair.reshape(B * L * L, 1, -1)
        templ = templ.permute(0, 2, 3, 1, 4).reshape(B * L * L, T, -1)
        out = self.attn(pair, templ, templ).reshape(B, L, L, -1)
        pair = pair.reshape(B, L, L, -1)
        pair = pair + out

        return pair, state


class MinimalIterBlock(nn.Module):
    """`IterBlock` (`Track_module.py:296-334`) with the `use_checkpoint`
    branch removed -- always the eager path."""

    def __init__(self, d_msa=256, d_pair=128, n_head_msa=8, n_head_pair=4,
                 use_global_attn=False, d_hidden=32, d_hidden_msa=None, p_drop=0.15,
                 SE3_param=None):
        super().__init__()
        if d_hidden_msa is None:
            d_hidden_msa = d_hidden
        # `None` sentinel instead of a `{...}` literal default -- see
        # minimal_rf_module.py's MinimalRoseTTAFoldModule.__init__ for why.
        SE3_param = SE3_param or {"l0_in_features": 32, "l0_out_features": 16, "num_edge_features": 32}
        self.msa2msa = MSAPairStr2MSA(
            d_msa=d_msa, d_pair=d_pair, n_head=n_head_msa, d_state=SE3_param["l0_out_features"],
            use_global_attn=use_global_attn, d_hidden=d_hidden_msa, p_drop=p_drop,
        )
        self.msa2pair = MSA2Pair(d_msa=d_msa, d_pair=d_pair, d_hidden=d_hidden // 2, p_drop=p_drop)
        self.pair2pair = PairStr2Pair(d_pair=d_pair, n_head=n_head_pair, d_hidden=d_hidden, p_drop=p_drop)
        self.str2str = Str2Str(
            d_msa=d_msa, d_pair=d_pair, d_state=SE3_param["l0_out_features"], SE3_param=SE3_param, p_drop=p_drop,
        )

    def forward(self, msa, pair, R_in, T_in, xyz, state, idx, motif_mask, cyclic_reses=None):
        rbf_feat = rbf(pairwise_distance(xyz[:, :, 1, :]))
        msa = self.msa2msa(msa, pair, rbf_feat, state)
        pair = self.msa2pair(msa, pair)
        pair = self.pair2pair(pair, rbf_feat)
        R, T, state, alpha = self.str2str(
            msa, pair, R_in, T_in, xyz, state, idx, motif_mask=motif_mask, cyclic_reses=cyclic_reses, top_k=0,
        )
        return msa, pair, R, T, state, alpha


class MinimalIterativeSimulator(nn.Module):
    """`IterativeSimulator` (`Track_module.py:336-477`), built from
    `MinimalIterBlock` instead of `IterBlock`. `use_checkpoint` is dropped
    entirely from the signature -- `IterativeSimulator.forward` itself
    never branched on it, it was only ever a pass-through to `IterBlock`.
    `motif_mask` is required rather than defaulted from `None`: every
    caller in this pipeline always provides one -- see
    `minimal_rf_module.py`'s module docstring for why relying on the
    original's own `None` fallback (`torch.zeros(L).bool()`, no
    `device=`) is actively wrong on GPU, not just redundant here."""

    def __init__(self, n_extra_block=4, n_main_block=12, n_ref_block=4,
                 d_msa=256, d_msa_full=64, d_pair=128, d_hidden=32,
                 n_head_msa=8, n_head_pair=4,
                 SE3_param_full=None,
                 SE3_param_topk=None,
                 p_drop=0.15):
        super().__init__()
        # `None` sentinel instead of a `{...}` literal default -- see
        # minimal_rf_module.py's MinimalRoseTTAFoldModule.__init__ for why.
        SE3_param_full = SE3_param_full or {"l0_in_features": 32, "l0_out_features": 16, "num_edge_features": 32}
        SE3_param_topk = SE3_param_topk or {"l0_in_features": 32, "l0_out_features": 16, "num_edge_features": 32}

        self.n_extra_block = n_extra_block
        self.n_main_block = n_main_block
        self.n_ref_block = n_ref_block

        self.proj_state = nn.Linear(SE3_param_topk["l0_out_features"], SE3_param_full["l0_out_features"])
        self.extra_block = nn.ModuleList([
            MinimalIterBlock(
                d_msa=d_msa_full, d_pair=d_pair, n_head_msa=n_head_msa, n_head_pair=n_head_pair,
                d_hidden_msa=8, d_hidden=d_hidden, p_drop=p_drop, use_global_attn=True,
                SE3_param=SE3_param_full,
            )
            for _ in range(n_extra_block)
        ])
        self.main_block = nn.ModuleList([
            MinimalIterBlock(
                d_msa=d_msa, d_pair=d_pair, n_head_msa=n_head_msa, n_head_pair=n_head_pair,
                d_hidden=d_hidden, p_drop=p_drop, use_global_attn=False,
                SE3_param=SE3_param_full,
            )
            for _ in range(n_main_block)
        ])
        self.proj_state2 = nn.Linear(SE3_param_full["l0_out_features"], SE3_param_topk["l0_out_features"])
        self.str_refiner = Str2Str(
            d_msa=d_msa, d_pair=d_pair, d_state=SE3_param_topk["l0_out_features"],
            SE3_param=SE3_param_topk, p_drop=p_drop,
        )

        self.proj_state = init_lecun_normal(self.proj_state)
        nn.init.zeros_(self.proj_state.bias)
        self.proj_state2 = init_lecun_normal(self.proj_state2)
        nn.init.zeros_(self.proj_state2.bias)

    def forward(self, seq, msa, msa_full, pair, xyz_in, state, idx, motif_mask, cyclic_reses=None):
        B, L = pair.shape[:2]

        R_in = torch.eye(3, device=xyz_in.device).reshape(1, 1, 3, 3).expand(B, L, -1, -1)
        T_in = xyz_in[:, :, 1].clone()
        xyz_in = xyz_in - T_in.unsqueeze(-2)

        state = self.proj_state(state)

        R_s, T_s, alpha_s = [], [], []
        for i_m in range(self.n_extra_block):
            R_in = R_in.detach()
            T_in = T_in.detach()
            xyz = einsum("bnij,bnaj->bnai", R_in, xyz_in) + T_in.unsqueeze(-2)
            msa_full, pair, R_in, T_in, state, alpha = self.extra_block[i_m](
                msa_full, pair, R_in, T_in, xyz, state, idx, motif_mask=motif_mask, cyclic_reses=cyclic_reses,
            )
            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)

        for i_m in range(self.n_main_block):
            R_in = R_in.detach()
            T_in = T_in.detach()
            xyz = einsum("bnij,bnaj->bnai", R_in, xyz_in) + T_in.unsqueeze(-2)
            msa, pair, R_in, T_in, state, alpha = self.main_block[i_m](
                msa, pair, R_in, T_in, xyz, state, idx, motif_mask=motif_mask, cyclic_reses=cyclic_reses,
            )
            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)

        state = self.proj_state2(state)
        for i_m in range(self.n_ref_block):
            R_in = R_in.detach()
            T_in = T_in.detach()
            xyz = einsum("bnij,bnaj->bnai", R_in, xyz_in) + T_in.unsqueeze(-2)
            R_in, T_in, state, alpha = self.str_refiner(
                msa, pair, R_in, T_in, xyz, state, idx, top_k=64, motif_mask=motif_mask, cyclic_reses=cyclic_reses,
            )
            R_s.append(R_in)
            T_s.append(T_in)
            alpha_s.append(alpha)

        R_s = torch.stack(R_s, dim=0)
        T_s = torch.stack(T_s, dim=0)
        alpha_s = torch.stack(alpha_s, dim=0)

        return msa, pair, R_s, T_s, alpha_s, state
