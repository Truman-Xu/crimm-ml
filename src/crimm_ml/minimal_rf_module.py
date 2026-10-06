"""A branch-free reimplementation of `rfdiffusion.RoseTTAFoldModel.
RoseTTAFoldModule`, specialized to exactly how it is called for
loop-building inference with `models/Base_ckpt.pt` (the same use case
`minimal_loop_sampler.py` strips the sampling loop *around* this model
for). `RoseTTAFoldModule` itself is left completely untouched -- this is a
separate module, built so that a `torch.compile`/export pipeline sees a
graph with no data-independent control flow left in it (every `if`
statement in the original `forward()`, and the config-time branches that
feed it, are Python-level facts here, not runtime tensor-shaped guards).

Every simplification below was checked against the actual call site this
pipeline reaches -- `SelfConditioning.sample_step`,
`rfdiffusion/inference/model_runners.py:803-869` (mirrored verbatim in
`minimal_loop_sampler.py:_sample_step`) -- and the config `Base_ckpt.pt`
itself carries (dumped straight from its `config_dict`), not assumed:

  - `msa_prev`/`pair_prev`/`state_prev` are passed as `None` at *every*
    call site in this repo -- there is no cross-call recycling anywhere in
    this pipeline, despite `RoseTTAFoldModule` owning a `Recycling`
    submodule; each call recycles from a freshly zeroed state. The
    `if msa_prev == None:` branch is always taken -> hardcoded to always
    zero-init, and the three `_prev` arguments are dropped from the
    signature (and, since they're never real, no longer returned either --
    the original returns them purely to let a caller thread them back in
    next call, which this pipeline never does).
  - `hasattr(self, 'timestep_embedder')` is **always** False, for *any*
    config, not just this one -- `timestep_embedder` is never assigned
    anywhere in this codebase. (`use_motif_timestep` is accepted by
    `RoseTTAFoldModule.__init__` but never used to build any such
    attribute, or anything else -- a dead constructor arg.) The block, and
    the `t`/`motif_mask`-derived `time_emb` it would have produced, are
    dropped entirely; `t` is otherwise unused anywhere in `forward()`, so
    it is dropped from the signature too.
  - `return_raw` and `return_full` are never set True by either
    `Sampler.sample_step` or `SelfConditioning.sample_step` -- only
    `return_infer=True` is ever passed. The `return_raw` early return, and
    the function's tail past the `return_infer` block (`c6d_pred`/
    `exp_pred`/full-trajectory `xyz` stack) are dead for this use case and
    removed, along with the `DistanceNetwork`/`ExpResolvedNetwork` heads
    that fed them -- unused, so not constructed here at all. (Their
    weights still exist in `Base_ckpt.pt`'s state dict; load with
    `strict=False`, or use `from_checkpoint` below, which already does.)
  - `is_frozen_residue = motif_mask if self.freeze_track_motif else
    torch.zeros_like(motif_mask).bool()`: `Base_ckpt.pt`'s own saved
    config has `freeze_track_motif: False` -- the value
    `Sampler.initialize()` actually loads at runtime
    (`assemble_config_from_chk` overwrites `base.yaml`'s placeholder with
    the checkpoint's own value) -- so the `else` branch is the only one
    ever taken for this checkpoint, and `freeze_track_motif` is dropped as
    a constructor arg (like `T`, `use_motif_timestep`, it was accepted by
    the original `__init__` but never referenced in its body beyond that
    one ternary). The caller no longer needs to supply `motif_mask` --
    this module builds the all-`False` ("nothing frozen") mask itself,
    directly on the right device. It deliberately does *not* lean on
    `IterativeSimulator`/`Str2Str`'s own `motif_mask=None` fallback
    (`rfdiffusion/Track_module.py:402-403`: `torch.zeros(L).bool()`, no
    `device=` arg) -- that silently produces a CPU tensor while every other
    tensor in this pipeline is on GPU, and the resulting device-mismatched
    mask was verified to *change* the simulator's output (~0.08 max abs
    diff in `pair`, amplifying further downstream) rather than raise. A
    latent bug in the original code's own default, not something to build
    on; building the mask explicitly on `xyz.device` sidesteps it.
  - `use_checkpoint` is never passed `True` at either call site --
    activation checkpointing trades compute for activation memory during
    *training*; both `sample_step` implementations run the whole model
    under `torch.no_grad()`, with no backward pass to save memory for.
    Dropped from this module's signature entirely, one level deeper than
    just calling the originals with `False`: `templ_emb`/`simulator` are
    now `minimal_track_module.MinimalTemplEmb`/`MinimalIterativeSimulator`,
    which don't accept `use_checkpoint` at all -- their own (and
    `TemplatePairStack`/`IterBlock`'s) `if use_checkpoint: ... else: ...`
    branches are removed at the source, not just defaulted away. See
    `minimal_track_module.py`'s module docstring for the full inventory.
  - `i_cycle`/`n_cycle` are accepted by the original `forward()` but never
    read anywhere in `RoseTTAFoldModel.py`, for *any* config or call site
    -- dropped.
  - `cyclic_reses` is always an all-`False` tensor at the one call site
    that ever populates it (`SelfConditioning.sample_step`'s
    `self.cyclic_reses`, itself hardcoded all-`False` whenever cyclic
    peptides aren't in play -- see `minimal_loop_sampler.py`'s
    `_sample_init`). Every place that actually reads it --
    `PositionalEncoding2D`'s cyclize block (`Embeddings.py:43-51`) and
    `get_seqsep`'s cyclic-edges block (`util_module.py:123-131`) -- only
    ever does anything when at least one position is actually cyclic
    (`cur_cyclize.sum() >= 2`/`cur_cyclic.sum() >= 2`); with an all-`False`
    mask both are unconditionally no-ops. `None` reproduces the exact same
    "nothing cyclic" behavior (verified: replaying a real call with
    `cyclic_reses=None` for `latent_emb` instead of the real all-`False`
    tensor gave a bit-exact match) while skipping that dead work entirely
    -- `numpy`/CPU round-trips (`find_breaks`) included. Dropped from this
    module's `forward()` signature; `None` is passed to `latent_emb`/
    `simulator` unconditionally.

Net effect: zero `if`/`hasattr` statements gated on a config value or a
call-site invariant remain in `forward()`.

Numerically cross-checked against the untouched `RoseTTAFoldModule` at a
single self-conditioning-off (t == T) step on `examples/input_pdbs/2KL8.pdb`
-- see the RFdiffusion fork's `crimm_integration/examples/verify_minimal_rf_module.py`. Result:
bit-exact (`0.0` max abs diff, all 7 output tensors).

Getting there required chasing down an apparent "non-determinism" first:
the very *first* call to either model in a fresh process -- `RoseTTAFoldModule`
or `MinimalRoseTTAFoldModule`, doesn't matter which -- takes a measurably
different path than every call after it (up to ~0.18 max abs diff in `pair`
against a second call with bit-identical inputs, growing from an initial
divergence to that size over 32 main blocks). Bisected with forward hooks
(capturing a real call's exact tensors, then replaying them standalone) to
rule out: the model being a non-deterministic function of its inputs (ruled
out -- replaying identical captured tensors through `simulator`, a single
`IterBlock`, the `SE3TransformerWrapper`, and the raw `dgl`-backed
`SE3Transformer` was bit-exact at every one of those levels in isolation);
input mutation between calls (ruled out -- every tensor fed to `simulator`
was verified bit-identical between two independent top-level calls, yet
their *outputs* still differed); and CUDA kernel-launch race conditions
(ruled out -- `CUDA_LAUNCH_BLOCKING=1` made no difference). What remained:
adding one throwaway warm-up call before anything else eliminates the
discrepancy completely -- every call from the second one on is bit-exact
with every other. This points to a one-time kernel-autotuning or JIT-
compilation step (most likely inside `dgl.ops.copy_e_sum`/`e_dot_v`, the
sparse graph-reduction primitives `SE3TransformerWrapper` uses under the
hood, or `e3nn`'s basis computation) that runs once per process and produces
a slightly different first result, not a persistent source of
irreproducibility. **Practical takeaway: always run one throwaway forward
pass through a freshly constructed/loaded model of this family before
timing, comparing, or relying on bit-reproducibility from it** -- both
`verify_minimal_rf_module.py` and any future benchmarking of either module
should do this warm-up call first. `MinimalRoseTTAFoldModule.warmup(L)`
below does exactly that (all-zero dummy tensors of the right shape for a
given sequence length -- kernel autotuning is keyed on tensor shape/dtype,
not values, so real data isn't needed), so callers don't have to
reconstruct the warm-up call's tensor shapes by hand.
"""
import torch
import torch.nn as nn
# `torch.einsum`, not `opt_einsum.contract` -- matching the patch applied
# throughout `crimm_ml/minimal_rf/`. dynamo cannot trace
# opt_einsum's pure-Python `contract`, and mixing the two here would make this
# module disagree with the rest of the package at the 1e-3 level. See
# `minimal_rf/__init__.py` for the rationale and measurements.
from torch import einsum

from crimm_ml.minimal_rf.AuxiliaryPredictor import LDDTNetwork, MaskedTokenNetwork
from crimm_ml.minimal_rf.Embeddings import Extra_emb, MSA_emb, Recycling

from crimm_ml.minimal_track_module import MinimalIterativeSimulator, MinimalTemplEmb

# Constructor kwargs `RoseTTAFoldModule` accepts but never uses for anything
# (not stored, or stored and read in exactly the one ternary this module
# hardcodes away -- see module docstring) -- stripped from `Base_ckpt.pt`'s
# saved `config_dict['model']` before it's passed to this class's __init__.
#: Keys `Base_ckpt.pt`'s saved model config carries that configure nothing.
#: `T`/`freeze_track_motif` are explained in the class docstring;
#: `use_motif_timestep` and the two `d_time_emb*` dims are all inputs to a
#: `timestep_embedder` that is never assigned anywhere in RFdiffusion -- the
#: only reads are `if hasattr(self, 'timestep_embedder')`, permanently False.
_DEAD_MODEL_KEYS = ("T", "use_motif_timestep", "freeze_track_motif",
                    "d_time_emb", "d_time_emb_proj")


class MinimalRoseTTAFoldModule(nn.Module):
    """`RoseTTAFoldModule` stripped to exactly the `return_infer=True`,
    `Base_ckpt.pt`, no-cross-call-recycling, no-motif-freezing,
    no-activation-checkpointing path `SelfConditioning.sample_step`
    actually exercises. See module docstring for what was removed, and
    why each removal is exact (not an approximation) for this use case.
    """

    def __init__(
        self,
        n_extra_block,
        n_main_block,
        n_ref_block,
        d_msa,
        d_msa_full,
        d_pair,
        d_templ,
        n_head_msa,
        n_head_pair,
        n_head_templ,
        d_hidden,
        d_hidden_templ,
        p_drop,
        d_t1d,
        d_t2d,
        SE3_param_full=None,
        SE3_param_topk=None,
        input_seq_onehot=False,
    ):
        super().__init__()
        # `{...}` literal defaults are a shared-mutable-object footgun for a
        # public constructor (harmless today since neither dict is ever
        # written to, but not worth relying on that staying true) -- `None`
        # sentinel + built fresh per call instead.
        SE3_param_full = SE3_param_full or {"l0_in_features": 32, "l0_out_features": 16, "num_edge_features": 32}
        SE3_param_topk = SE3_param_topk or {"l0_in_features": 32, "l0_out_features": 16, "num_edge_features": 32}

        self.d_t1d = d_t1d
        self.d_t2d = d_t2d
        d_state = SE3_param_topk["l0_out_features"]
        self.latent_emb = MSA_emb(
            d_msa=d_msa, d_pair=d_pair, d_state=d_state,
            p_drop=p_drop, input_seq_onehot=input_seq_onehot,
        )
        self.full_emb = Extra_emb(
            d_msa=d_msa_full, d_init=25, p_drop=p_drop, input_seq_onehot=input_seq_onehot,
        )
        self.templ_emb = MinimalTemplEmb(
            d_pair=d_pair, d_templ=d_templ, d_state=d_state, n_head=n_head_templ,
            d_hidden=d_hidden_templ, p_drop=0.25, d_t1d=d_t1d, d_t2d=d_t2d,
        )
        self.recycle = Recycling(d_msa=d_msa, d_pair=d_pair, d_state=d_state)
        self.simulator = MinimalIterativeSimulator(
            n_extra_block=n_extra_block, n_main_block=n_main_block, n_ref_block=n_ref_block,
            d_msa=d_msa, d_msa_full=d_msa_full, d_pair=d_pair, d_hidden=d_hidden,
            n_head_msa=n_head_msa, n_head_pair=n_head_pair,
            SE3_param_full=SE3_param_full, SE3_param_topk=SE3_param_topk, p_drop=p_drop,
        )
        self.aa_pred = MaskedTokenNetwork(d_msa)
        self.lddt_pred = LDDTNetwork(d_state)

    def forward(self, msa_latent, msa_full, seq, xyz, idx, t1d, t2d, xyz_t, alpha_t):
        B, N, L = msa_latent.shape[:3]

        # cyclic_reses=None: exact, cheaper equivalent of the all-`False`
        # tensor this pipeline always uses -- see module docstring.
        msa_latent, pair, state = self.latent_emb(msa_latent, seq, idx, None)
        msa_full = self.full_emb(msa_full, seq, idx)

        # Recycling always starts from zero: no call site in this pipeline
        # ever carries msa_prev/pair_prev/state_prev across calls.
        msa_prev = torch.zeros_like(msa_latent[:, 0])
        pair_prev = torch.zeros_like(pair)
        state_prev = torch.zeros_like(state)
        msa_recycle, pair_recycle, state_recycle = self.recycle(seq, msa_prev, pair_prev, xyz, state_prev)
        msa_latent = msa_latent.clone()
        msa_latent[:, 0] = msa_latent[:, 0] + msa_recycle.reshape(B, L, -1)
        pair = pair + pair_recycle
        state = state + state_recycle

        pair, state = self.templ_emb(t1d, t2d, alpha_t, xyz_t, pair, state)

        # All-False ("nothing frozen") mask, since freeze_track_motif is
        # always False for Base_ckpt.pt -- built explicitly. MinimalIterativeSimulator
        # requires motif_mask (no `None` fallback): the original's own fallback
        # (Track_module.py:402-403: `torch.zeros(L).bool()`, no `device=`) is a
        # CPU tensor even on a GPU pipeline -- verified experimentally that the
        # resulting device-mismatched mask changes simulator output (~0.08 max
        # abs diff in `pair`, growing from there) rather than raising. Building
        # the mask ourselves on the right device sidesteps that bug entirely.
        motif_mask = torch.zeros(L, dtype=torch.bool, device=xyz.device)
        msa, pair, R, T, alpha_s, state = self.simulator(
            seq, msa_latent, msa_full, pair, xyz[:, :, :3], state, idx,
            motif_mask=motif_mask, cyclic_reses=None,
        )

        logits_aa = self.aa_pred(msa)
        lddt = self.lddt_pred(state)

        xyz_out = einsum(
            "bnij,bnaj->bnai", R[-1], xyz[:, :, :3] - xyz[:, :, 1].unsqueeze(-2)
        ) + T[-1].unsqueeze(-2)

        nbin = lddt.shape[1]
        bin_step = 1.0 / nbin
        lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=lddt.dtype, device=lddt.device)
        pred_lddt = nn.Softmax(dim=1)(lddt)
        pred_lddt = torch.sum(lddt_bins[None, :, None] * pred_lddt, dim=1)

        return msa[:, 0], pair, xyz_out, state, alpha_s[-1], logits_aa.permute(0, 2, 1), pred_lddt

    @torch.no_grad()
    def warmup(self, L, device=None):
        """Runs one throwaway forward pass on all-zero dummy tensors shaped
        for a sequence of length `L`, and discards the result.

        Two uses. First, the very first call in a fresh process takes a
        measurably different path than every call after it (one-time kernel
        autotuning), so bit-reproducible output requires calling this once
        first. Second, on a compiled model this pass *is* the compile -- do it
        here rather than inside a sampling loop. With `dynamic=True`, the
        default, one warmup covers every length, so `L` need not match the
        target you then run. Values don't matter for either purpose -- only
        shape/dtype do -- so all-zero dummy tensors are enough. Shapes are exactly what `minimal_loop_sampler.py`'s
        `_preprocess` builds for a real call, at this L:
          - msa_latent (1,1,L,48), msa_full (1,1,L,25), seq (1,L,22)
          - xyz, xyz_t: (1,L,27,3), (1,1,L,27,3) -- atom14 padded to the
            full 27-atom slots the template-coordinate pathway expects
          - idx (1,L); t1d (1,1,L,d_t1d); t2d (1,1,L,L,d_t2d);
            alpha_t (1,1,L,30) -- 10 torsions x (sin,cos,mask), anatomical
            and independent of `d_t1d`/`d_t2d`/every other config knob
        Restores whatever `self.training` was before the call.
        """
        device = device or next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            msa_latent = torch.zeros(1, 1, L, 48, device=device)
            msa_full = torch.zeros(1, 1, L, 25, device=device)
            seq = torch.zeros(1, L, 22, device=device)
            xyz = torch.zeros(1, L, 27, 3, device=device)
            idx = torch.arange(L, device=device)[None]
            t1d = torch.zeros(1, 1, L, self.d_t1d, device=device)
            t2d = torch.zeros(1, 1, L, L, self.d_t2d, device=device)
            xyz_t = torch.zeros(1, 1, L, 27, 3, device=device)
            alpha_t = torch.zeros(1, 1, L, 30, device=device)
            self(msa_latent, msa_full, seq, xyz, idx, t1d, t2d, xyz_t, alpha_t)
        finally:
            self.train(was_training)

    # Compiling this model lives in `crimm_ml/compilation.py`, and
    # shipping the compiled result lives in `crimm_ml/compiled_model.py`.
    # Neither is a property of the network, and keeping them out of this class
    # means the model can be built, loaded and verified without torch.compile
    # being involved at all:
    #
    #     from crimm_ml import compiled_model
    #     model, _ = MinimalRoseTTAFoldModule.from_checkpoint(ckpt, device="cpu")
    #     compiled_model.load_and_compile(model, warmup_L=L)

    @classmethod
    def from_checkpoint(cls, ckpt, device=None, **overrides):
        """Build directly from a `Base_ckpt.pt`-shaped checkpoint dict:
        `ckpt['config_dict']['model']` (minus `_DEAD_MODEL_KEYS`) plus
        `d_t1d`/`d_t2d` from `ckpt['config_dict']['preprocess']` --
        exactly the values `Sampler.load_model`
        (`rfdiffusion/inference/model_runners.py:252-270`) computes for
        `RoseTTAFoldModule`. Loads `ckpt['model_state_dict']` with
        `strict=False`: the checkpoint's `c6d_pred`/`exp_pred` heads have
        no counterpart here (see module docstring) and show up as
        `unexpected_keys`, by design -- nothing should ever be `missing`.
        Returns `(model, load_result)`. The model is eager; to compile it see
        `crimm_ml.compiled_model.load_and_compile`.
        """
        model_conf = {
            k: v for k, v in ckpt["config_dict"]["model"].items() if k not in _DEAD_MODEL_KEYS
        }
        model_conf["d_t1d"] = ckpt["config_dict"]["preprocess"]["d_t1d"]
        model_conf["d_t2d"] = ckpt["config_dict"]["preprocess"]["d_t2d"]
        model_conf.update(overrides)

        model = cls(**model_conf)
        if device is not None:
            model = model.to(device)
        model = model.eval()
        load_result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        assert not load_result.missing_keys, load_result.missing_keys
        return model, load_result
