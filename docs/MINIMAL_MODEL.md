# How the minimal model was built

*[README](../README.md) · [Compiling](COMPILING.md) · [Distributing](DISTRIBUTING.md) · **Minimal model***

> A research log, written in the RFdiffusion fork (branch
> `crimm-integration-onnx`) before this package existed. Paths are as they
> were there: `crimm_integration/` is now `src/crimm_ml/`, and the scripts
> it cites stay in the fork unless they are in `tools/export/`.

The engineering record behind `minimal_rf_module.py` and `minimal_rf/`: what was
stripped out of RFdiffusion's `RoseTTAFoldModule` and why, what broke along the
way, and how the model came to be DGL-free and compilable.

This is the *why it looks like this* document. For how to build and run it, see
**[COMPILING.md](COMPILING.md)**; for how it gets shipped, see
**[DISTRIBUTING.md](DISTRIBUTING.md)**.

Read it in order — each section is a consequence of the one before it. The short
version:

1. Strip the model to the branches this one use case actually reaches.
2. That exposed a real device bug and a one-time GPU cold-start effect.
3. `torch.compile` then failed on two blockers, one of which was DGL.
4. Removing DGL fixed the ceiling, unpinned torch, and made export possible.

> **On the numbers in this document.** This is a record of how the model got
> here, so its measurements are the ones taken at the time — mostly torch 2.4 on
> an RTX 2080 Ti, before DGL was removed. They are accurate for the state of the
> code being described and are kept for that reason. Where a later measurement
> superseded one, the text says so. **For current guidance always use
> [COMPILING.md](COMPILING.md)**, which is measured on torch 2.14 on CPU
> throughout.

---

## Stripping the model itself for `torch.compile`/distribution

Everything above strips the *sampling loop* around the model
(`Sampler`/`SelfConditioning`). `minimal_rf_module.py` does the same thing
one level down, to `rfdiffusion.RoseTTAFoldModel.RoseTTAFoldModule` itself:
`MinimalRoseTTAFoldModule` is a from-scratch reimplementation of its
`forward()` with **every `if`/`hasattr` statement resolved away**, using the
same evidence-before-editing approach as the rest of this package — each
removal is justified against the actual call site
(`SelfConditioning.sample_step`, `model_runners.py:803-869`) and
`Base_ckpt.pt`'s own saved config, not assumed. `RoseTTAFoldModule` itself
is untouched.

What's resolved, and why (see the module's docstring for the full
line-by-line justification):

| Branch in the original `forward()` | Resolves to | Why |
|---|---|---|
| `if msa_prev == None:` (recycling) | always zero-init | every call site in this repo passes `msa_prev`/`pair_prev`/`state_prev=None`; there is no cross-call recycling in this pipeline |
| `if hasattr(self, 'timestep_embedder'):` | dead, removed | `timestep_embedder` is never assigned anywhere in this codebase, for *any* config |
| `if return_raw:` / tail after `return_infer` (`c6d_pred`/`exp_pred`) | dead, removed | only `return_infer=True` is ever passed; the `DistanceNetwork`/`ExpResolvedNetwork` heads that fed the dead tail aren't constructed at all |
| `is_frozen_residue = motif_mask if self.freeze_track_motif else ...` | always the `else` (all-`False`) | `Base_ckpt.pt`'s own saved config has `freeze_track_motif: False` |
| `use_checkpoint` (gradient checkpointing) | removed at the source, not just defaulted `False` | never passed `True` at either call site — both run under `torch.no_grad()`, nothing to save backward-pass memory for |
| `t`, `i_cycle`, `n_cycle` | dropped from the signature | unused anywhere in `forward()` outside the dead `timestep_embedder` block (`i_cycle`/`n_cycle` are unused everywhere, for any config) |

`use_checkpoint` doesn't stop at `RoseTTAFoldModule.forward` — `Templ_emb`,
`TemplatePairStack`, and `IterBlock` (all in `rfdiffusion/Embeddings.py`/
`Track_module.py`) each carry their own `if use_checkpoint: ... else: ...`
branch, always taking the `else`. `minimal_track_module.py` provides
branch-free replacements — `MinimalTemplEmb`, `MinimalTemplatePairStack`,
`MinimalIterBlock`, `MinimalIterativeSimulator` — that remove those
branches at the source instead of just calling the originals with `False`.
Everything without a `use_checkpoint` branch of its own
(`MSAPairStr2MSA`, `MSA2Pair`, `PairStr2Pair`, `Str2Str`, and every
attention/feedforward building block) is imported and reused directly,
unmodified — confirmed by grepping `Attention_module.py`/`util_module.py`
for `use_checkpoint`/`checkpoint.`: no matches. `__init__`, submodule
types, and parameter names/shapes are otherwise identical to the
originals, so `Base_ckpt.pt`'s state dict loads into these classes exactly
the way it loads into the originals (same keys, same shapes) — re-running
`verify_minimal_rf_module.py` after this change still reports `missing_keys: 0`
and the same bit-exact result as before.

(`TemplateTorsionStack`, `Embeddings.py:168-197`, also has a
`use_checkpoint` branch but is genuinely dead code for *any* config —
`Templ_emb.__init__` never instantiates it, `self.tor_stack = ...` is
commented out in the original source — so there's nothing to strip there.)

### A real bug this surfaced: `motif_mask=None`'s device

`IterativeSimulator`/`Str2Str`'s own fallback for `motif_mask=None`
(`Track_module.py:402-403`) is `torch.zeros(L).bool()` — no `device=` arg,
so it's a **CPU** tensor even when everything else in the pipeline is on
GPU. Passing `None` through (instead of building the all-`False` mask
ourselves) was verified to *change* the simulator's output (~0.08 max abs
diff in `pair`) rather than raise a device-mismatch error. `crimm_integration`
can't fix this in the original file (out of scope, and not `crimm`'s bug to
begin with — it's RFdiffusion's own), so `MinimalRoseTTAFoldModule` just
builds the mask itself, explicitly, on the right device.

### A bigger finding: a one-time GPU "cold start" effect — resolved with a warm-up call

Validating this module needed a real answer to "does it produce the same
output as the original," and the naive check — run both once, diff — kept
failing by amounts too large to be floating-point noise (~0.01–0.19 in
`pair`). It first looked like the *original*, unmodified `RoseTTAFoldModule`
was itself non-deterministic: called twice in a row with bit-identical
inputs, it didn't produce bit-identical output.

Bisected with forward hooks — capture a real call's exact tensors at a
given point, then replay those exact tensors standalone — to rule out, in
order: the model being a non-deterministic function of its inputs (ruled
out: replaying identical captured tensors through `simulator`, a single
`IterBlock`, `SE3TransformerWrapper`, and the raw `dgl`-backed
`SE3Transformer` was bit-exact at every one of those levels checked in
isolation); silent input mutation between calls (ruled out: every tensor
fed to `simulator` was confirmed bit-identical between two independent
top-level calls, yet their outputs still differed); and CUDA kernel-launch
race conditions (ruled out: `CUDA_LAUNCH_BLOCKING=1` made no difference).

What actually explained it: **the very first call to either model in a
fresh process takes a measurably different path than every call after
it.** Adding one throwaway warm-up call before anything else eliminates the
discrepancy completely — every call from the second one on is bit-exact
with every other, for both models, on all 7 output tensors. This points to
a one-time kernel-autotuning or JIT-compilation step (most likely inside
`dgl.ops.copy_e_sum`/`e_dot_v`, the sparse graph-reduction primitives
`SE3TransformerWrapper` uses, or `e3nn`'s basis computation) that runs once
per process, not a persistent source of irreproducibility.
`crimm_integration/examples/verify_minimal_rf_module.py` now warms up both
models with one throwaway call first, then asserts exact equality (not
"within noise") on everything after — and gets it: `0.0` max abs diff, all
7 output tensors, both models. **Practical takeaway: always run one
throwaway forward pass through a freshly constructed/loaded model of this
family before timing, comparing, or relying on bit-reproducibility from
it** — a cheap habit, and standard practice for `torch.compile` pipelines
anyway (which need a warm-up call to trigger compilation regardless).

### Further stripping opportunities — resolved

All five items previously listed here have been addressed:

- **Warm-up call, baked in.** `MinimalRoseTTAFoldModule.warmup(L, device=None)`
  runs one throwaway forward pass on all-zero dummy tensors shaped for a
  sequence of length `L` (kernel autotuning is keyed on shape/dtype, not
  values, so real data isn't needed) and discards the result. Callers no
  longer need to hand-construct the warm-up call's tensor shapes themselves.
- **Mutable-default-arg smell, fixed.** `SE3_param_full`/`SE3_param_topk`/
  `SE3_param` now default to `None` and are filled in with a fresh dict
  inside `__init__` (`minimal_rf_module.py` and `minimal_track_module.py`),
  rather than sharing one dict literal object across every instance.
- **`cyclic_reses` folded away.** Dropped from `MinimalRoseTTAFoldModule.forward`'s
  signature entirely; `None` is passed to `latent_emb`/`simulator`
  unconditionally. Both places that ever read `cyclic_reses`
  (`PositionalEncoding2D`'s cyclize block and `get_seqsep`'s cyclic-edges
  block) only do anything when at least one residue is actually cyclic, so
  an all-`False` tensor makes them no-ops anyway, and `None` skips entering
  those loops at all. Verified bit-exact equivalence before and after.

  **Correction to an earlier claim in this file.** This bullet used to say
  that passing `None` also skipped the `numpy`/CPU round-trip
  (`find_breaks`) inside `get_seqsep`. That was wrong, and measuring it
  later proved it wrong. `find_breaks` and the `chainids` construction run
  *before* the `if cyclic is not None:` guard, in both `get_seqsep`
  (`util_module.py:116`) and `PositionalEncoding2D.forward`
  (`Embeddings.py:33`) — so they execute on every call regardless, and only
  their *consumer* is skipped. That dead round-trip turned out to cost
  about 10% of the eager forward pass and to be the single largest source
  of `torch.compile` graph breaks; see the `torch.compile` section below.
- **Verified at `T=50`** (`Base_ckpt.pt`'s actual trained/production value,
  not just `T=15`, the beta schedule's minimum) — `verify_minimal_rf_module.py`
  now runs the full bit-exact check at both, and both come back bit-exact.
- **`torch.compile` — attempted, then actually made to work.** The first
  attempt hit two blockers; both have since been resolved, and resolving
  them exposed two further problems that mattered more than either. Net
  result on this cluster: 1.59x faster than the model as shipped. See below.

### `torch.compile`: what actually happens when you try it

Three scripts cover this. `examples/compile_minimal_rf_module.py` was the
first attempt and recorded the two blockers it hit.
`examples/compile_se3_probe.py` is the follow-up that got past them, and —
more usefully — found that neither was the main thing standing between this
model and a fast compiled forward pass. `examples/compile_cache_probe.py`
then asks whether the compile is affordable at all once it is cached; see
"Compiling by default, and caching it" below for the answer, which is "only
for batch runs".

**The fixes described below are no longer applied at runtime.** They are baked
into the source of `crimm_integration/minimal_rf/`, each marked with a
`# PATCH` comment. The probe now measures the result rather than producing it,
and its phases run independently:

```bash
srun --partition=gpu2080 --gres=gpu:1 --time=01:00:00 bash -lc '
    module load gcc/15.2.0
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate RFDiffusion
    python crimm_integration/examples/compile_se3_probe.py env baseline split breaks compile'
```

#### Blocker 1: the compiler — fixed by a module load

TorchInductor's generated C++ uses the C++20 `[[unlikely]]` attribute; this
cluster's default `g++` is 8.5.0, which does not parse it (`error: expected
identifier before '[' token`). GCC only accepts that attribute from about
version 9. `module load gcc/15.2.0` before invoking `python` resolves it
outright — no `conda install` and no risk to the pinned `torch`/`dgl` CUDA
wheels. Everything below assumes that module is loaded; the probe's `env`
phase checks the compiler version and warns loudly if it is not.

#### Blocker 2: dynamo cannot trace the SE(3)-Transformer — fenced off, not removed

The failure surfaces as a shape error deep inside the radial MLP
(`se3_transformer/model/layers/convolution.py:118`): `a and b must have same
reduction dim, but got [5700, 32] X [33, 32]`. That layer is
`nn.Linear(33, 32)`, and 33 is `num_edge_features + 1`. The extra `+1` is the
edge *length*, which `get_populated_edge_features` (`transformer.py:50-59`)
concatenates onto the edge features by reading `graph.edata['rel_pos']` off
the `DGLGraph`. Under tracing that concatenation never reaches the graph, so
the MLP is handed 32 features instead of 33. 5700 is exactly `L*(L-1)` for
`L=76`, i.e. the full-graph edge count.

Two distinct things make this region untraceable:

- `make_topk_graph`/`make_full_graph` (`util_module.py:135,159`) derive the
  edge list with `b, i, j = torch.where(cond)`. Single-argument `torch.where`
  is `nonzero`, a data-dependent output-shape op: the edge count depends on
  coordinate *values*, not on any input shape. Feeding the same tensors with
  jittered coordinates changes the edge count while every input shape stays
  identical, which is precisely what dynamo has no symbolic expression for.
- The result is packed into a `dgl.DGLGraph`, an opaque Python object that
  then carries tensors on `.edata` and is consumed by `dgl.ops.*`.

**The workaround is a fence, not a rewrite.** The model touches DGL through
exactly three functions, so decorating each with `torch._dynamo.disable` turns
an un-traceable region into a clean graph break at a boundary we picked. In
`minimal_rf` these are source decorators on `util_module.make_full_graph`,
`util_module.make_topk_graph` and `SE3_network.SE3TransformerWrapper.forward`.

With that in place dynamo traces the whole model without error. The SE(3)
part stays eager, which is a real cost — see the time split below.

#### What the fence exposed: two problems that mattered more

With tracing working, `torch._dynamo.explain` reported **936 graph breaks**.
Almost none of them were DGL.

**1. `einsum` is `opt_einsum.contract`, not `torch.einsum`.** Every module in
the model gets the name from `from opt_einsum import contract as einsum`
(`Attention_module.py:5`, `util_module.py:5`, and transitively via
`import *`). `opt_einsum.contract` is a third-party Python function; dynamo
cannot trace into it and breaks at *every single call site* — roughly 26 per
`IterBlock`, times 36 blocks. Pointing those module-level names at
`torch.einsum` instead cut breaks from 936 to 194.

This is safe. All 19 distinct subscript expressions in the model are plain
two-operand contractions with no ellipses, where there is only one possible
contraction path, so the path-search `opt_einsum` provides is irrelevant.
Three representative contractions were checked op-by-op and come back
**bit-identical**. It is also simply faster in plain eager, with no
compilation involved at all.

One trap worth recording: this swap must be applied to *every* module in the
package at once. Leaving `minimal_rf_module.py` and `minimal_track_module.py`
on `opt_einsum` while the vendored modules used `torch.einsum` made the two
halves of `verify_minimal_rf_module.py` disagree at the 1e-3 level, and the
bit-exactness check caught it.

**2. A dead `numpy` round-trip on the hot path.** `get_seqsep`
(`util_module.py:105`) and `PositionalEncoding2D.forward`
(`Embeddings.py:25`) both compute chain IDs like this:

```python
breaks   = find_breaks(idx.squeeze().cpu().numpy())    # GPU -> CPU sync
chainids = np.zeros_like(idx.squeeze().cpu().numpy())  # GPU -> CPU sync
for i, b in enumerate(breaks): chainids[b:] = i + 1    # python loop
chainids = torch.from_numpy(chainids).to(idx.device)   # CPU -> GPU
```

and then read `chainids` **only** inside `if cyclic is not None:`. For loop
building there is no cyclic peptide, so all of it is computed and discarded —
40 times per forward pass (once per `Str2Str`), each with two forced
GPU-to-CPU synchronisations. `find_breaks` uses `np.where`, which dynamo
lowers to `aten.nonzero`, so it was also the single largest remaining source
of graph breaks: **78 of the 194**.

Guarding the block behind the same `is not None` test that already gates its
only consumer is behaviour-preserving by construction. The outputs are
**bit-exact**, and it alone is worth about 10% of eager runtime.

**3. Dynamo silently gave up on most of the model.** With both source fixes
in, a naive `torch.compile(model)` still spent 25+ minutes compiling and
emitted 14 `torch._dynamo hit config.cache_size_limit (8)` warnings — after
which dynamo stops compiling those frames and runs them **eager**, so the
"compiled" model was mostly interpreted. The cause is torch 2.4's default
`inline_inbuilt_nn_modules = False`, which bakes each module instance's
parameters into its graph as constants. The trunk is 36 structurally
identical `IterBlock`s, so every block became its own specialisation. Setting
`torch._dynamo.config.inline_inbuilt_nn_modules = True` makes parameters
graph inputs, and all 36 blocks share one compiled artifact: compile time
drops to 57s, `unique_graphs` is 15, and there are zero cache-limit
fallbacks. (This is the default from torch 2.5 onward.)

#### Where the time actually goes

From the `split` phase (eager, with per-call synchronisation, so treat these
as shares rather than absolute times):

| region | share of forward | calls |
| --- | --- | --- |
| `Str2Str` total | 67.5% | 40 |
| ...of which `SE3TransformerWrapper` | 42.3% | 40 |
| ...of which DGL graph construction | 5.0% | 40 |
| MSA/pair trunk + embeddings | 32.5% | — |

The fenced SE(3) region is about two thirds of the runtime and stays eager,
which caps what compilation can win here. Its share is now *larger* than when
first measured (63.3%), precisely because the two source patches made the
trunk faster while leaving this region untouched.

#### Result

`L=76`, RTX 2080 Ti, `T=15`, first reverse-diffusion step, median of 3 after
warm-up, all from a single run of `compile_se3_probe.py` so the numbers are
mutually comparable:

| configuration | ms / forward | graph breaks |
| --- | --- | --- |
| stock `rfdiffusion`, eager | 448 | 936 |
| `minimal_rf`, eager (both source patches) | 389 | 14 |
| `minimal_rf` + `inline_inbuilt_nn_modules` + `torch.compile` | **330** | 14 |

**1.36x end-to-end**, of which 1.15x comes from the source patches before any
compilation at all. Compilation produces 15 unique graphs; all 14 remaining
breaks are the DGL fence (`make_*_graph` and `SE3TransformerWrapper.forward`).

That compile was timed at 48s here, but treat that as a **partly-warmed**
number: this script's `compile` phase runs after `breaks`, which has already
traced the whole model twice via `dynamo.explain`. A cold process pays 92-98s —
see the next section, where the compile is the only thing each process does.

Treat these as approximate: repeated runs on different nodes of the same
partition vary by 10-15%, and an earlier run measured 516/423/324 for the
same three configurations. The ratios are stable; the absolute numbers are
not.

The break count is only 14 **with `inline_inbuilt_nn_modules = True`**.
Without it, dynamo specialises each of the 36 identical `IterBlock`s
separately and counts the same break site once per block, giving 149. The
probe's `breaks` phase reports both.

Accuracy: the `find_breaks` guard is bit-exact. The `torch.einsum` swap moves
outputs by at most ~7e-4 relative against stock RFdiffusion, and compilation
adds a similar amount. Coordinates stay within ~5e-4 Angstrom and `pred_lddt`
within ~2e-5. The noisiest output is `alpha_s[-1]`, the torsion prediction,
which has been seen to differ by up to 4e-3 absolute on values bounded by 1.
This is floating-point reassociation amplified through 36 sequential blocks,
not a semantic change.

#### Compiling by default, and caching it

`compilation.py` + `compiled_model.py` +
`examples/prime_compile_cache.py` cover this. The short version is that
compiling is **deliberately not the default**, and the arithmetic below is why.

Two things stop a compile from being reused by the next process, and neither is
fixed out of the box on torch 2.4:

- `torch._inductor.config.fx_graph_cache` is `False` (it becomes the default in
  torch 2.6). With it off there is no cross-process cache at all, wherever the
  cache directory points.
- `TORCHINDUCTOR_CACHE_DIR` defaults to `/tmp/torchinductor_$USER`, and `/tmp`
  is **node-local** on this cluster. A cache written by a job on `gollum154` is
  invisible to the next job, which will usually land somewhere else.
  `compilation.configure()` moves it to `~/.cache/crimm_rfdiffusion/torchinductor`,
  on NFS home. Triton's own kernel cache already defaults to `~/.triton/cache`
  and needs no help.

Measured on one RTX 2080 Ti at `L=76`, each compile in its own process
(`examples/compile_cache_probe.py`):

| compile | wall time | FX-graph cache |
| --- | --- | --- |
| dynamo only (`backend="eager"`) | 24.4 s | n/a |
| cold, static shapes | 98.1 s | 3 hit / 13 miss |
| **warm, static shapes** | **30.4 s, 29.6 s** | 16-17 hit / **0 miss** |
| cold, `dynamic=True` | 201.8 s | 0 hit / 14 miss |
| warm, `dynamic=True` | 103.7 s | 14 hit / 0 miss |

Caching recovers about 70% of the compile. What is left is essentially the
24.4 s dynamo floor: bytecode analysis, guard construction and AOTAutograd
tracing are redone in full by every process, and no setting changes that on
2.4 — cross-process caching of that layer only arrives with `AOTAutogradCache`
(2.6) and `torch.compiler.save_cache_artifacts` (2.7).

**Neither the cache nor its location costs anything on a cold compile.**
Repeating the static compile with `TORCHINDUCTOR_CACHE_DIR` on node-local
`$SLURM_LOCAL_SCRATCH` gave 91.5 s cold and 28.8 s warm, against 98.1 s and
29.6-30.4 s on `$HOME`; turning the FX-graph cache off entirely gave 92.5 s
(with 0 hits and 0 misses, confirming it was really off). All within run-to-run
variance on a shared node. So enabling the cache is free, and there is no
reason to trade the fleet-wide location for a node-local one.

This also means the 48 s compile reported in the previous section is **not**
what a cold process pays — it was measured after `dynamo.explain` had already
traced the model twice in the same process. Budget 92-98 s cold, ~30 s warm.

**The break-even is the point.** Compilation saves ~60 ms on a ~390 ms forward
pass, so even a warm 30 s compile needs ~500 forward passes to pay for itself —
about **10 designs at T=50**, or 33 at T=15. One gap fill at `num_designs=1` is
50 forward passes and loses badly. Hence: compile for batch design runs, not
for one-off gap fills.

Two practical notes:

- **`dynamic=True` is now the default, and the advice above it was wrong.**
  Everything in this section down to here was measured on torch 2.4 on GPU,
  when the model still had 936 graph breaks and `torch.compile` covered only a
  third of it. Re-measured on torch 2.14 on CPU after the DGL removal
  (`examples/dynamic_shape_probe.py`, 2KL8 at L = 76, 80, 100, 57 in one
  process), `dynamic=True` wins on every axis it used to lose on:

  | mode | dynamo compiles across 4 lengths | per forward, steady state |
  | --- | --- | --- |
  | `dynamic=False` | 57 | 0.46 – 5.55 s |
  | `dynamic=None` (auto) | 36 | 0.40 – 1.37 s |
  | **`dynamic=True`** | **18** | **0.30 – 0.67 s** |

  One compile, then L=80, L=100 and L=57 each cost **zero** further graphs —
  including L=57, below `make_topk_graph`'s `top_k=64` where `min(top_k, L)`
  looks like it must specialise and does not. Verified across L = 70 … 160 (a
  2.3× span) on one graph, at a steady 1.4 – 2.4× over eager. The old "slightly
  slower per forward" claim does not survive: dynamic is now the *faster* and
  much more uniform of the two.

  The remaining honest caveat is cross-process, not in-process: a fresh process
  redoes dynamo's tracing, which the inductor cache does not hold. Static is
  13.0 s for a length you primed and 70.1 s for one you did not; dynamic is a
  flat ~90 s at any length. Prime specific lengths with `--static` if you can
  enumerate them; otherwise take the flat cost. See COMPILING.md, "Changing the
  protein — does `L` force a recompile?".
- **Prime the cache in its own job.** `examples/prime_compile_cache.py` needs
  only the checkpoint — no PDB, no contig, no diffuser — because the compiled
  graph depends only on shapes and dtypes. Pay the cold compile there, off the
  critical path, then every later run starts warm. The cache key includes the
  device's compute capability, so priming on `gpu2080` does not prime
  `ada6000`; prime on the partition you will run on.

How to actually turn it on:

```bash
# once per GPU architecture, in its own short job
srun --partition=gpu2080 --gres=gpu:1 --time=00:30:00 bash -lc '
    module load gcc/15.2.0
    source $(conda info --base)/etc/profile.d/conda.sh && conda activate RFDiffusion
    python crimm_integration/examples/prime_compile_cache.py --device cpu 76.
```

```python
# then, in the run itself
from crimm_integration import compile_cache
from crimm_integration.minimal_rf_module import MinimalRoseTTAFoldModule

compilation.configure()                    # cache off /tmp, blob capture on
model, _ = MinimalRoseTTAFoldModule.from_checkpoint(ckpt, device="cuda",
                                                    compile=True)
model.warmup(L, device="cuda")             # pays the (now warm) compile here
```

`compilation.compile_model()` calls `configure()` itself if you forget, and
`disable_compile()` puts the eager `forward` back for A/B measurement.

Accuracy is unaffected by any of this: compiled-vs-eager worst relative
difference ranged 1.7e-4 to 1.3e-3 across the six runs above, always dominated
by `alpha_s[-1]` (the torsion head, already the noisiest output — the run that
touched 1.3e-3 was 1.7e-3 absolute on values bounded by ~1.27). Coordinates
stayed within 6.4e-4 Angstrom and `pred_lddt` within 3.1e-5 throughout. This is
the same floating-point-reassociation level as the source patches themselves.

#### Removing DGL: done, and what it cost

The section above ended by calling DGL removal "a bounded piece of work, not
the architectural dead end the first attempt concluded it was." It has now been
done, and the estimate was right about the shape but wrong about the size: the
dependency was *two* packages, not one.

`crimm_integration/minimal_rf/se3/` is a vendored, pure-torch
SE(3)-Transformer. `torch` and `numpy` are the only runtime imports left in the
model path — verified by importing and running the SE(3) stack with `dgl`,
`e3nn`, `se3_transformer`, `opt_einsum` **and** `rfdiffusion` all blocked by an
import hook.

| upstream | replacement | agreement |
| --- | --- | --- |
| `graph.edges()`, `graph.edata['rel_pos']` | `EdgeGraph` NamedTuple of tensors | exact |
| `dgl.ops.copy_e_sum` | `index_add_` at dst | bit-exact |
| `dgl.ops.e_dot_v` | `(e * v[dst]).sum(-1)` | ~1e-6 |
| `dgl.ops.edge_softmax` | segment softmax over dst | ~1e-8 |
| `e3nn.o3.wigner_3j` | 244 baked constants (`_wigner_3j_data.py`) | bit-exact |
| `e3nn.o3.spherical_harmonics` | vendored polynomial | bit-exact |

e3nn turned out to be the easier half. `wigner_3j` is a pure function of three
integers — upstream already `lru_cache`d it — so it is a table, and a 7 KB
sparse table ships instead of a dependency. The spherical harmonics are a
closed-form polynomial in x, y, z, copied verbatim.

`Base_ckpt.pt` sets `num_degrees=2`, so `max_degree` is **1**, not the 3 the
class default suggests. Only degrees 0–2 of the harmonics and J ∈ {0,1,2} are
ever touched.

**Verification is layered so a failure localises** — `examples/verify_no_dgl.py`,
four phases: the DGL primitives against DGL itself, the bases against
e3nn/NVIDIA, a whole `SE3TransformerWrapper` against NVIDIA's, and the full
model against stock `rfdiffusion`. The decisive one is the third: **bit-exact,
`0.0e+00`, on both SE3 configurations `Base_ckpt.pt` uses**, with its state
dict loading `strict=True` (44 tensors, 0 missing, 0 unexpected). Parameter
names were preserved deliberately so this comparison is possible at all.

**One genuine regression, and it is not a rounding difference.** DGL's
`copy_e_sum` is a segment reduction over a sorted CSR and is bit-reproducible
on CUDA. `index_add_` uses atomics, so accumulation order varies between calls:
1.2e-5 max self-difference over 5 calls of one op, which the 36-block trunk
amplifies to ~6e-5 relative in `pair` and ~4e-4 Å in output coordinates.
Irrelevant to structure quality, but the model is no longer bit-reproducible
run-to-run by default, which it previously was. `torch.use_deterministic_algorithms(True)`
restores it exactly (0.0 over 5 calls) — torch routes `index_add_` to a
sort-based kernel — and both verification scripts now set it, which is what
returns `verify_minimal_rf_module.py` to BIT-EXACT at T=15 and T=50. Sorting
edges by destination does *not* help; the atomics remain either way.

#### What it bought

| | with DGL (fenced) | DGL removed |
| --- | --- | --- |
| graph breaks | 14 | **17** |
| dynamo `calls_captured` | 1188 | **2389** |
| `unique_graphs` | 15 | 18 |
| runtime stuck in eager | ~67% (the SE(3) region) | only `nonzero` |
| runtime deps of the model | torch, numpy, dgl, e3nn, opt_einsum | **torch, numpy** |

The break count going 14 → 17 is not a regression: those 14 were the DGL fence
truncating the graph early, and the three extra boundaries come with roughly
**twice as many ops captured**. What that is worth, measured in one run on an
RTX 2080 Ti at L=76 (`compile_se3_probe.py env baseline split breaks compile`):

| configuration | ms / forward | vs stock |
| --- | --- | --- |
| stock `rfdiffusion`, eager | 449.4 | 1.00x |
| `minimal_rf` eager, DGL fenced (previous section) | 388.6 | 1.16x |
| `minimal_rf` eager, DGL removed | 332.8 | 1.35x |
| `minimal_rf` compiled, DGL fenced (previous section) | 329.9 | 1.36x |
| **`minimal_rf` compiled, DGL removed** | **205.8** | **2.18x** |

Compilation now buys 1.63x over patched eager, where behind the fence it bought
only 1.18x — the fenced region was the ceiling, and removing it lifted the
ceiling rather than just shaving the trunk. Compile time rose to 121 s cold
(more graph to compile); the caching arithmetic in the previous section still
applies, with a correspondingly better payoff per forward pass.

Accuracy is unchanged in character: `minimal_rf` vs stock worst relative
difference 1.25e-4, compiled vs eager 1.75e-4.

The only fence left is `util_module._edge_index_from_cond`, wrapping the single
`torch.where(cond)` whose output shape depends on coordinate values rather than
on any input shape. Everything else, the whole SE(3) region included, now
traces.

Two consequences beyond speed:

- **DGL was what pinned this environment to torch 2.4.** The `dgl==2.4.0+cu121`
  wheel is built against a specific torch minor version, which is why
  `env_setup.md` warns against bumping torch. That constraint is gone.
- **Export is now conceivable at all.** A `DGLGraph` is an opaque Python object
  that `torch.export` cannot represent; an `EdgeGraph` is a pytree of tensors.

#### Toward a torch-free distributable model

**Update: done, verified, and ONNX is now the distribution recommendation.**
The export is shape-dynamic and numerically correct, and after optimisation ONNX
Runtime runs at **1.0-1.1x PyTorch eager on CPU** (0.99 s against 0.90 s per
forward at L=76, 8 threads, dedicated node) and about **2x faster than PyTorch
on GPU**. The sampler around it is pure NumPy, so `pip install crimm[ml]` needs
no PyTorch at all.

An earlier version of this update reported ONNX Runtime as 2.2-3.1x *slower*
and recommended PyTorch CPU-only instead. That figure came from the unoptimised
fixed-`L` export, measured on a busy shared login node; both causes are
explained in [DISTRIBUTING.md](DISTRIBUTING.md), *What changed*. Full numbers,
the export bugs found along the way, and step-by-step instructions are in
[COMPILING.md](COMPILING.md).

The original analysis follows.

The goal is for `pip install crimm` to bring a working model with no CUDA
toolkit, no compiler and no hardware questions. Assessment of the options:

- **ONNX + ONNX Runtime — the intended target.** The CPU wheel is a
  self-contained manylinux binary with no CUDA; `onnxruntime-gpu` pulls its
  CUDA runtime as ordinary pip wheels. ONNX has a native `NonZero` op, so the
  data-dependent edge construction that defeats dynamo may export cleanly — the
  17 graph breaks are a dynamo limitation, not a format one.
- **AOTInductor** — rejected. It emits a `.so` that still links libtorch at
  runtime, which is the dependency being shed.
- **ExecuTorch** — plausible for CPU (XNNPACK, small runtime), but its CUDA
  story is immature, and it needs a break-free export just as AOTInductor does.

torch 2.4's TorchScript-based ONNX exporter fails on this model with an
internal `list index out of range` after 35 s, which is why the export work
moved to a modern torch — now possible precisely because DGL no longer pins it.

CPU is viable on its own: **1647 ms/forward at L=76 on 20 threads**, about 4x
an RTX 2080 Ti, i.e. ~82 s for one design at T=50.
