# Shipping the model to users

*[README](../README.md) · [Compiling](COMPILING.md) · **Distributing** · [Minimal model](MINIMAL_MODEL.md)*

The goal this serves, stated at the outset of the project: installing `crimm`
should give a working loop-building model **without the user configuring
anything or knowing their hardware**.

## The decision

**Ship the ONNX export as a separate `crimm-ml` distribution, installable as
`pip install crimm[ml]`, with the model file fetched on first use.** No PyTorch
on the user's machine at all.

The runtime dependency set is numpy, scipy, omegaconf and onnxruntime — about
15 MB of wheels on top of crimm — and the sampler around the model is pure
NumPy, so nothing in the install path asks the user about hardware, CUDA, or a
compiler. The 275 MB model cannot go in a wheel (PyPI allows 100 MB per file),
so it downloads once, checksummed, into a user cache; a container image bakes it
in instead.

Five things drove the choice:

1. **It is as fast as PyTorch eager on CPU.** At L=76, 8 threads on a dedicated
   node: **0.99 s** per forward against PyTorch's 0.90 s — 1.1x, across
   L=57-100 (COMPILING.md §4, *Known limitations*). That is the single number
   that changed the recommendation; see *What changed* below.
2. **It is faster than PyTorch on GPU.** 0.16 s per step against 0.35 s on an
   RTX 2080 Ti, with final coordinates agreeing to 6.2e-6 Å.
3. **It is length-agnostic.** The export is shape-dynamic: one file covers every
   protein, with no recompile and no per-length artifact (COMPILING.md §4,
   *What made `L` dynamic*).
4. **The install resolves cleanly.** onnxruntime's wheel carries loose pins and
   no `nvidia-*` packages. `pip install torch` on Linux pulls a 900 MB wheel and
   15 pinned `nvidia-*` distributions unless the user passes a custom index URL
   — and a package's own metadata *cannot* express that, because `+cpu` local
   versions may not be published to PyPI. In an environment that already holds
   rdkit, biopython, scipy and pandas, that difference decides it.
5. **It has no second implementation to maintain.** The ONNX file is produced
   from the same `minimal_rf` source by a script in the repo, and verified
   against PyTorch before shipping: a full T=50 trajectory agrees to 3.8e-6 Å
   (`tools/export/verify_numpy_sampler.py onnx`).

### The distribution shape

| Piece | Where it lives |
|---|---|
| Engine: `models/`, `numpy_rf/`, `minimal_rf/`, progress, asset fetch | `crimm-ml` on PyPI (import `crimm_ml`), its own repo, depends on `crimm` |
| Chain-facing glue: gap selection, contig building, residue rebuild | `crimm` itself, imported lazily so `import crimm` stays light |
| The 275 MB `.onnx` + `.onnx.data` pair, and the `schedules/*.npz` | A GitHub Release, mirrored to Zenodo for a citable DOI; cached under `platformdirs.user_cache_dir` |
| Export and graph-rewrite tooling (needs torch + onnx) | the same repo, behind an `[export]` extra |
| The service | the same repo, behind a `[service]` extra; container image ships the assets baked in |

Redistributing the weights is explicitly permitted: RFdiffusion's licence is BSD
and its text covers the model weights (reproduced in `NOTICE`). Attribution travels in
the release notes and a `NOTICE` file.

### What changed, and why the earlier recommendation was wrong

This document previously recommended shipping a `torch.compile` cache blob on
top of a CPU-only PyTorch wheel, and rejected ONNX on speed with this table:

| | ONNX Runtime | PyTorch CPU-only |
|---|---|---|
| speed (16 threads) | 2209 ms | 710 ms |

Both numbers are now known to be unrepresentative, for two independent reasons:

- **The ONNX model was unoptimised.** Four changes — matmul attention,
  cheaper Transpose layout, a `F.pad`+`cat` basis construction, and
  `SkipLayerNormalization`/`FusedMatMul` graph edits — took L=76 from 1.56 s to
  0.99 s (COMPILING.md §4).
- **The measurement was taken on a shared login node under load.** Both runtimes
  ran ~1.8x slower there and ONNX Runtime was hit harder, which inflated the gap
  on top of the real one. Every current figure is from a dedicated `srun` node.

The 3x gap that justified requiring PyTorch was therefore about 1.1x. Separately,
the blob mechanism itself stopped working (below). Two of the old objections to
ONNX have also simply been fixed: the export is no longer fixed-`L`, and the
no-torch sampler means ONNX is a complete path rather than a model file needing a
PyTorch harness around it.

---

## The alternatives, and why not

### `torch.compile` cache blob — the previous recommendation

`torch.compiler.save_cache_artifacts()` (torch 2.7+) bundles a compile into one
portable blob that can ride along in a wheel, so a user's first run does not pay
for it. Measured in a fresh process with the on-disk cache **wiped**, so only
the blob could help ("setup" is everything before the first real forward:
loading the blob, `torch.compile`, and one warmup pass):

| configuration | blob | `torch.compile()` | warmup | **setup** |
| --- | --- | --- | --- | --- |
| no blob (control) | — | 1.17 s | 204.4 s | **205.5 s** |
| blob + `caching_precompile=True` | 25.1 MB | 27.7 s | 4.0 s | **31.9 s** |

It worked, it degraded gracefully on a miss, and it stayed shape-dynamic. Three
things rule it out as the default:

1. **It does not currently load (torch 2.14).** Loading a `caching_precompile`
   blob aborts inside dynamo with
   `AssertionError: Name '__builtins_dict___3' already exists in scope`,
   reproducibly, in a fresh process, on an empty cache and a warm one alike, via
   the sampler and via the low-level API. It was observed working once — the
   build job that created the blob, 6.7 s startup — and has not reproduced since.
   With `caching_precompile=False` it does not crash but buys nothing: 206 s
   against 220 s cold. `compiled_model.load_and_compile` therefore catches the
   failure and falls back to eager rather than taking the caller's run down.
2. **It still requires PyTorch**, with a version-pinned blob, one per torch
   version and not hardware-neutral — it survived a different CPU generation
   (3.5 s vs 8.4 s of warmup) but a shipping matrix was never decided.
3. **It only pays from the second design onward.** A single T=50 design is ~40 s
   eager against ~53 s blob-compiled, because setup is not amortised at one
   design.

The mechanism stays available for batch work on a machine that already has
PyTorch: `crimm_ml.compilation` and `crimm_ml.compiled_model`, plus the
fork's `crimm_integration/examples/megacache_probe.py`, which wipes the
inductor cache so only the blob can be responsible for a speedup.

> **Measure the whole of setup, not just the warmup.** With `caching_precompile`
> the work moves into the `torch.compile()` call rather than the first forward.
> An earlier version of this document reported 3.6 s and a 55x speedup by timing
> only the warmup; the honest figure is **6.4x**.

### AOTInductor

Rejected on portability, not performance. Its artifact is tied to the exact
PyTorch that built it — in **both** directions, failing at `dlopen` before any
model code runs — and "standalone" mode currently removes only the link-time
dependency, not the runtime one (COMPILING.md §3). The contrast that decided it
against the blob applies here too: a stale cache blob costs a recompile; a
mismatched AOTInductor artifact is an unrecoverable load error on the user's
machine.

Worth re-checking on each PyTorch release: if it ever produces a genuinely
standalone artifact, it would beat ONNX on speed with the same "no PyTorch"
property.

### TensorFlow

Considered and rejected without building anything. A reimplementation means
porting ~5,400 lines of SE(3)-Transformer, scatter aggregation and top-k graph
building, remapping the checkpoint, and then maintaining a second implementation
forever. The converter route (`onnx2tf`; `onnx-tf` is archived) would have to
survive `NonZero`, `TopK`, `ScatterND`, `ScatterElements` with reduction,
`Einsum` and a symbolic sequence length — exactly the constructs such tools
handle worst, and the CUDA `ReduceL2` bug found here (COMPILING.md §4, *ONNX
Runtime graph edits*) shows how those failures present: silently wrong
numbers, no error.

On installation it is the *worse* answer, which was the main thing it was being
considered for: TensorFlow pins numpy, protobuf and ml-dtypes, and its CUDA wheel
runs to hundreds of MB, against onnxruntime's ~15 MB CPU wheel and loose pins.

TPUs do not help either: they want XLA with static shapes and large batches,
while this is a batch-1, 50-step sequential loop whose edge count comes from
`NonZero`. The measured lever for speed is a GPU, where ONNX already wins.

---

## Open items

- **The 275 MB asset download has no fallback mirror yet.** GitHub Release
  first, Zenodo second; the Zenodo deposit is not made.
- **`crimm[ml]` → `crimm-ml` → `crimm` is a dependency cycle.** pip resolves it,
  but it needs proving on TestPyPI before a real release.
- **`import crimm` currently pulls in rdkit and nglview**, which defeats the
  small-install argument above. Fix proposed in `import-fix.md` in the crimm
  repo.
- **Memory grows as L³** (432 MB at L=300) with no enforced cap in the library;
  the service imposes one (COMPILING.md §4).
- **AOTInductor re-evaluation** on each PyTorch release (COMPILING.md §3).
