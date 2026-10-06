"""A self-contained, patched subset of RFdiffusion.

Everything in this package is a copy of a file from the `rfdiffusion/` package
at the repository root, with imports rewritten to be relative so that nothing
here imports `rfdiffusion` at all. The point is distribution: `crimm_ml`
can be lifted out and shipped without carrying the whole RFdiffusion repo, and
without the risk that a future change to `rfdiffusion/` silently alters model
behaviour underneath it.

What was copied
---------------
The transitive closure of what `minimal_loop_sampler.py`, `minimal_rf_module.py`
and `minimal_track_module.py` actually import, after pruning everything that
turned out to be unreachable -- 15 modules, about 5,380 lines
(plus `se3/`):

    Attention_module.py
    AuxiliaryPredictor.py
    Embeddings.py
    RoseTTAFoldModel.py
    SE3_network.py
    Track_module.py
    chemical.py
    contigs.py
    diffusion.py
    igso3.py
    inference/utils.py
    kinematics.py
    potentials/manager.py
    util.py
    util_module.py

`RoseTTAFoldModel.py` is the one exception: nothing in the pipeline imports it.
It is the reference `examples/verify_minimal_rf_module.py` compares the stripped
model against, and its own docstring says so.

Since the sampler became NumPy (2026-09-17), it imports only `chemical.py`,
`contigs.py`, `inference/pdb.py` and `potentials/manager.py` from here, none of
which import torch; the rest of its logic is the port in
`crimm_ml/numpy_rf/`. The torch versions of `diffusion.py`, `igso3.py`,
`inference/utils.py` and `kinematics.py` stay as the reference that
`examples/verify_numpy_sampler.py` checks the port against, and the network
still uses `util.py` and `util_module.py`.

What was pruned, and why it was safe
------------------------------------
About 1,600 lines went, each checked to be unreachable before removal rather
than assumed: `scoring.py` in full plus `util.py`'s LJ/hbond parameter tables
(built at import, never read by anything); `BlockAdjacency`/`Target`/
`ss_from_contig` (scaffold-guided only); `potentials/potentials.py` in full and
most of `manager.py` (`guiding_potentials` is `null` and unsettable here, so
`Denoise` always short-circuits); and a dozen unused helpers across
`kinematics.py`, `igso3.py`, `util.py` and `chemical.py`. The end-to-end
trajectory is bit-identical before and after, and
`verify_minimal_rf_module.py` still reports BIT-EXACT at T=15 and T=50.

What was deliberately NOT copied
--------------------------------
`inference/model_runners.py` (1214 lines), `inference/symmetry.py` and
`model_input_logger.py`. These entered the dependency graph only through
`sampler_selector` in `inference/utils.py`, which dispatches to the
general-purpose CLI samplers. `MinimalLoopSampler` drives the model directly,
so that function and its import were removed and the three modules fell out.

The patches
-----------
Every deviation from the original is marked with a `# PATCH` comment naming the
upstream file, so a future diff against `rfdiffusion/` stays readable. There are
five, all established by measurement rather than assumption -- the evidence is
in `docs/MINIMAL_MODEL.md`, and reproducible with
the RFdiffusion fork's `crimm_integration/examples/compile_se3_probe.py`:

1. **`torch.einsum` instead of `opt_einsum.contract`** (`Attention_module.py`,
   `Embeddings.py`, `RoseTTAFoldModel.py`, `util_module.py`). `opt_einsum`'s
   `contract` is plain Python that dynamo cannot trace, so it broke the compiled
   graph at all ~26 call sites per block. Every expression here is a two-operand
   contraction, so no contraction-path search is being given up. Also ~10%
   faster in eager.

2. **The dead `find_breaks` round-trip is guarded** (`util_module.get_seqsep`,
   `Embeddings.PositionalEncoding2D.forward`). Both built `chainids` via
   `.cpu().numpy()` on every call but read it only under `if cyclic is not
   None`. Now it is computed only when it is used. Bit-exact; worth ~10% of
   eager runtime and 78 graph breaks.

3. **DGL and e3nn removed outright.** `se3/` is a vendored, pure-torch copy of
   NVIDIA's `se3_transformer`: DGL's edge list, per-edge vectors and three
   scatter ops became plain tensor operations, e3nn's Wigner 3j symbols became
   a baked constant table, and its spherical harmonics were vendored verbatim.
   Parameter names are unchanged, so `Base_ckpt.pt` still loads `strict=True`,
   and the rewritten `SE3TransformerWrapper` is **bit-exact** against the
   DGL/e3nn original (`examples/verify_no_dgl.py`).

   This superseded an earlier workaround that fenced the three DGL entry points
   with `@torch._dynamo.disable`. One fence remains, on
   `util_module._edge_index_from_cond`: `torch.where(cond)` is `nonzero`, whose
   output shape depends on coordinate values rather than on any input shape.
   Everything else -- including the whole SE(3) region, about two thirds of the
   forward pass -- now traces. Graph breaks went 936 -> 17.

4. **`sampler_selector` and its `model_runners` import removed**
   (`inference/utils.py`), as described above.

5. **`torch.cuda.amp.autocast` -> `torch.amp.autocast('cuda', ...)`**
   (`Track_module.py`). The old spelling is deprecated and emits a
   FutureWarning on every import. Identical semantics -- and with
   `enabled=False` both are no-ops on CPU -- so this is behaviour-preserving on
   every device. This package targets torch versions that support the portable
   compile blob, so the deprecated alias buys nothing.

Caveat on numerics
------------------
Patches 2, 4 and 5 are exactly behaviour-preserving. Patch 1 is not *bit*-exact
end to end: individual einsum calls agree bit-for-bit, but different reduction
orders in a few contractions accumulate through 36 sequential blocks to a
worst-case relative difference of about 2e-4 against stock RFdiffusion --
coordinates move by ~5e-4 Angstrom. `examples/verify_minimal_rf_module.py`
checks against that tolerance and explains it.

To compile
----------
The fences above are necessary but not sufficient. Also required:

    module load gcc/15.2.0          # before python; 8.5.0 cannot build
                                    # TorchInductor's C++20 output
    torch._dynamo.config.inline_inbuilt_nn_modules = True   # default from 2.5

Without the second, dynamo specialises each of the 36 identical IterBlocks
separately, blows through `cache_size_limit`, and silently falls back to eager.
"""
