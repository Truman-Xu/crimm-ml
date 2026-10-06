# Compiling and running the loop-building model

*[README](../README.md) · **Compiling** · [Distributing](DISTRIBUTING.md) · [Minimal model](MINIMAL_MODEL.md)*

> Written in the RFdiffusion fork (branch `crimm-integration-onnx`) and moved
> here with the code. Tools it uses are in `tools/export/`; paths under
> `examples/` or `crimm_integration/examples/` are research probes that stay
> in that fork. The export tools need an RFdiffusion checkout with
> `models/Base_ckpt.pt`: set `RFDIFFUSION_DIR` (default `../RFdiffusion`).

How to build this model, make it fast, and check it is still correct. For how it
gets *shipped* to a user, see **[DISTRIBUTING.md](DISTRIBUTING.md)**; for why the
model looks the way it does, see **[MINIMAL_MODEL.md](MINIMAL_MODEL.md)**.

Three different things get called "compiling" here. They solve different
problems and are not alternatives so much as layers:

| | `torch.compile` | AOTInductor | ONNX export |
|---|---|---|---|
| Produces | a faster in-process model | a `.so` compiled ahead of time | a standalone `.onnx` file |
| Still needs PyTorch? | yes | yes (libtorch) | **no** — only `onnxruntime` |
| Speed | **~2x** over eager, at every length | same kernels as `torch.compile` | about the same as PyTorch eager on CPU (0.95-1.11x its time) |
| Setup per process | 205 s cold, **32 s** from a shipped cache blob | none at runtime | 4–26 s to load the session |
| Portable across torch versions? | yes (source) | **no** — hard `dlopen` failure | yes |
| Status | **verified, and what ships** | evaluated, not adopted | verified correct at any `L`, but slower |

If you want the model faster today, use `torch.compile` (§2). If you are
packaging it, read DISTRIBUTING.md — the short version is that the compile is
shipped as a cache blob rather than as a compiled binary.

---

## 0. Environments

There are two, deliberately.

**`RFDiffusion`** — the existing env. torch 2.4 + dgl + e3nn. Needed only to
compare against *stock* RFdiffusion; the model itself no longer uses dgl or e3nn.

**`crimm-export`** — torch 2.14 + onnx + onnxruntime, no dgl. Used for export.

```bash
conda create -n crimm-export python=3.11
conda activate crimm-export
pip install torch numpy scipy omegaconf onnx onnxruntime onnxscript
```

Removing DGL is what made this second env possible: the `dgl==2.4.0+cu121`
wheel is built against one specific torch minor version, and that is what
pinned the project to torch 2.4.

Everything below is run from the repo root (`/home/ziqiaoxu/RFdiffusion`).

---

## 1. Check the model is intact

Before compiling anything, confirm the DGL-free model still matches the
original. Four phases, ordered so a failure tells you *where* it broke:

```bash
srun --partition=gpu2080 --gres=gpu:1 --time=00:30:00 bash -lc '
  source $(conda info --base)/etc/profile.d/conda.sh && conda activate RFDiffusion
  cd /home/ziqiaoxu/RFdiffusion
  python crimm_integration/examples/verify_no_dgl.py --device cuda'
```

Expect `ops / basis / se3 / model` all **PASS**, with `se3` reporting
`0.000e+00` — the rewritten SE(3)-Transformer is bit-exact against NVIDIA's
DGL/e3nn original.

---

## 2. `torch.compile` — faster, still needs PyTorch

### Two things must be set, or it silently does nothing useful

```bash
module load gcc/15.2.0     # TorchInductor emits C++20; the default g++ 8.5.0 cannot build it
```

```python
from crimm_ml import compilation
compilation.configure()    # moves the inductor cache off node-local /tmp, and
                           # turns on caching_precompile so the compile can later
                           # be saved as a portable blob (DISTRIBUTING.md)
```

`compilation.configure()` is called for you by `compile_model()`; call it
directly only if you want a non-default cache directory.

### Using it

Compiling lives outside the model class, in two modules: `compilation` does the
compiling, `compiled_model` saves and loads the result. The model itself knows
nothing about either.

```python
from crimm_ml import compiled_model
from crimm_ml.minimal_rf_module import MinimalRoseTTAFoldModule

model, _ = MinimalRoseTTAFoldModule.from_checkpoint(ckpt, device="cpu")
compiled_model.load_and_compile(model, warmup_L=L)   # blob if present, else compile
```

`MinimalLoopSampler` has no compile option; it takes the network as a loaded
model object, and the usual alternative to eager mode is an ONNX model (§4).
To compile a PyTorch model anyway, compile its module before handing it over:

```python
from crimm_ml import compiled_model
from crimm_ml.models import load_model
from crimm_ml.minimal_loop_sampler import MinimalLoopSampler

model = load_model("models/Base_ckpt.pt")
compiled_model.load_and_compile(model.module)   # blob if present, else compile
sampler = MinimalLoopSampler(pdb_lines, contig, model, T=50)
```

Compiling does not always pay: measured at T=15, eager was 2.3 s init + 11.4 s
sampling against 6.7 s + 8.9 s compiled. Compiling pulls ahead as the design
count and `T` rise. Outputs agree to 3.9e-05 A.

The lower-level form, if you do not want the blob:

```python
from crimm_ml import compilation
compilation.compile_model(model)      # dynamic shapes by default
model.warmup(L)                       # the compile happens here, not in your loop
compilation.disable_compile(model)    # back to eager, for A/B measurement
```

Both require a torch new enough to support the portable blob; older versions
raise a `RuntimeError` naming what is missing rather than silently paying a full
compile in every process.

### Changing the protein — does `L` force a recompile?

No. `compilation.compile_model()` defaults to `dynamic=True`, which makes the sequence
length symbolic: **one compile serves every target length.**

Measured on CPU / 16 threads under torch 2.14, running 2KL8 at L = 76, 80, 100
and 57 in one process (`examples/dynamic_shape_probe.py`):

| mode | dynamo compiles across 4 lengths | per forward, steady state |
| --- | --- | --- |
| `dynamic=False` | 57 | 0.46 – 5.55 s |
| `dynamic=None` (auto) | 36 | 0.40 – 1.37 s |
| **`dynamic=True`** | **18** | **0.30 – 0.67 s** |

`dynamic=True` compiles once and then absorbs every other length with **zero**
further graphs — including L=57, which is below `make_topk_graph`'s `top_k=64`,
where `min(top_k, L)` looks like it must specialise and does not. A single
dynamic graph was verified across L = 70 … 160, a 2.3× span, with no recompiles
and a steady 1.4 – 2.4× speedup over eager at every length.

Accuracy is unaffected. Against eager at L=76, on the three outputs the sampler
consumes: `xyz_out` 4.6e-04 Å, `alpha` 2.4e-03, `pred_lddt` 3.8e-05 — the same
as `dynamic=False` gives, and far inside the 1e-2 gate `verify_onnx_model.py`
uses. Eager repeated against itself is bit-exact, so those are real
reassociation differences, just negligible ones.

Check it on your own inputs:

```bash
python crimm_integration/examples/dynamic_shape_probe.py --mode dynamic --threads 16
```

**Across processes it is a genuine trade-off, not a free win.** A fresh process
must redo dynamo's tracing, which the inductor cache does not hold:

| first forward in a fresh process | `dynamic=False` | `dynamic=True` |
| --- | --- | --- |
| cold, empty cache | 86.7 s | 187.5 s |
| warm cache, **same** `L`=76 | **13.0 s** | 87.9 s |
| warm cache, **new** `L`=110 | 70.1 s | 90.0 s |

So if you can enumerate your lengths ahead of time, `--static` priming of each
one is cheaper per process (13 s). If you cannot — the usual case when the
target changes — `dynamic=True` costs a flat ~90 s at *any* length, needs no
guessing, and never grows the cache.

**`torch._dynamo.mark_dynamic` on the input tensors does not work here**, and
the reason is worth knowing. The model still graph-breaks at the `nonzero`
fence in `util_module._edge_index_from_cond`, and every break hands dynamo fresh
intermediate tensors carrying concrete shapes. Marking only the nine entry
tensors leaves those static, and the `torch.cat` at `Track_module.py:264` then
unifies a symbolic `pair` with a constant-76 `neighbor`:

```
ConstraintViolationError: Constraints violated (L['idx'].size()[1])!
  - You marked L['idx'].size()[1] as dynamic but your code specialized it
    to be a constant (76).
```

`dynamic=True` covers post-break intermediates too; nothing narrower does.

### Is it worth it?

That depends entirely on whether you pay for the compile, and there are now three
regimes:

| how the compile is paid for | setup per process | worth compiling? |
| --- | --- | --- |
| cold, nothing cached | ~205 s | only for long batch runs |
| warm inductor cache, same `L` | ~13 s (warmup only) | usually yes |
| **shipped cache blob** (DISTRIBUTING.md) | **~32 s** | from ~2 designs onward |

Compiling buys roughly **2x** on the forward pass at every length tested
(L = 70 … 160): 0.42 s against 0.79 s at L=76 on 16 threads. One design at T=50
is 50 forwards, so eager costs ~40 s and blob-compiled ~53 s — **eager still
wins at one design**, and compiling pulls ahead from about the second.

Time the whole of setup, not just the warmup: with `caching_precompile` the work
moves into the `torch.compile()` call itself (27.7 s) rather than the first
forward (4.0 s), so a warmup-only measurement understates it by ~28 s. Compiling
stays an explicit call rather than something `from_checkpoint` does for you.

### Pay the compile once, offline

```bash
srun --partition=gpu2080 --gres=gpu:1 --time=00:30:00 bash -lc '
  module load gcc/15.2.0
  source $(conda info --base)/etc/profile.d/conda.sh && conda activate RFDiffusion
  cd /home/ziqiaoxu/RFdiffusion
  python crimm_integration/examples/prime_compile_cache.py --device cpu 76'
```

One length is enough: `compile_model()` defaults to `dynamic=True`, so the graph
it builds is shape-generic and the `L` you prime at need not match what you run.
Pass several lengths only with `--static`, which bakes `L` into the kernels.

The cache lives in `~/.cache/crimm_rfdiffusion/torchinductor` (on `$HOME`,
because `/tmp` is node-local here). Its key includes the device, so **prime on
the hardware you will actually run on** — priming on `gpu2080` does not prime
`ada6000`, and neither primes CPU.

To turn that cache into something shippable rather than machine-local, see
[DISTRIBUTING.md](DISTRIBUTING.md).

### Measure it yourself

```bash
srun --partition=gpu2080 --gres=gpu:1 --time=01:30:00 bash -lc '
  module load gcc/15.2.0
  source $(conda info --base)/etc/profile.d/conda.sh && conda activate RFDiffusion
  cd /home/ziqiaoxu/RFdiffusion
  python crimm_integration/examples/compile_se3_probe.py env baseline split breaks compile'
```

---

## 3. AOTInductor — compiled ahead of time, but pinned to one PyTorch

`torch.compile` does its work inside the running process. AOTInductor
(`torch._inductor.aoti_compile_and_package`) instead emits a `.so` at build
time, so a user's process does no tracing and no codegen at all. That is exactly
what a distributable artifact wants — which is why it was evaluated seriously,
and why the way it fails matters.

### It is not portable across PyTorch versions

AOTInductor (`torch._inductor.aoti_compile_and_package`) is the option that
removes tracing at runtime entirely, and is worth exploring for that reason. But
its artifact is **tied to the exact PyTorch that built it**, in both directions.
Tested with a two-layer MLP, exported with a dynamic batch dim, built in one env
and loaded in the other:

| built with | loaded on | result |
| --- | --- | --- |
| 2.14.0 | 2.4.0 | `undefined symbol: aoti_torch_new_tensor_handle` |
| 2.4.0 | 2.14.0 | `undefined symbol: c10::impl::cow::materialize_cow_storage(c10::StorageImpl&)` |

Both fail at `dlopen`, before any model code runs. Backwards fails because
AOTInductor's stable C shim *grows* over releases — `aoti_torch_new_tensor_handle`
does not exist in 2.4's libtorch. Forwards fails worse: the 2.4-built artifact
needs a mangled **C++** internal symbol, so it was never confined to the stable C
API at all. The `.pt2` container is also not readable by 2.4, which predates
`aoti_load_package` and only had `torch._export.aot_load` (deprecated by 2.14).

The practical consequence for `pip install crimm`: an AOTInductor artifact means
**pinning a narrow torch range in the package metadata** and rebuilding the
artifact per supported version and per architecture. That is a real maintenance
cost, and it is the one thing the ONNX route genuinely buys — runtime and
version independence — at roughly PyTorch-eager speed, as measured in §4.

Reproduce with `examples/aoti_portability_probe.py` (`build` in one env, `load`
in the other). Note also that building on torch 2.4 needs binutils >= 2.32 for
`objcopy --set-section-alignment`; RHEL 8's system binutils is 2.30 and fails.

### What the artifact actually depends on

`DT_NEEDED` of a default-built artifact (torch 2.14, CPU):

```
libtorch.so  libtorch_cpu.so  libgomp.so.1  libstdc++.so.6
libm.so.6  libgcc_s.so.1  libpthread.so.0  libc.so.6  ld-linux-x86-64.so.2
```

So **libtorch is not the only dependency**. It also needs the OpenMP runtime and
a libstdc++ new enough for the compiler that built it — the artifact above
requires `GLIBCXX_3.4.30`, which RHEL 8's system libstdc++ does not have, so it
fails to load on the very machine that built it unless the gcc module is loaded.
Weights are inside the `.so` by default (`package_constants_in_so=True`).

The libtorch SONAMEs are **unversioned** — 2.4 and 2.14 both ship
`libtorch.so` / `libtorch_cpu.so`. Two versions therefore cannot coexist on one
loader search path by name, and two libtorch copies in one process would clash
over duplicated global registries regardless. One libtorch per process.

**Can libtorch be baked into the artifact?** Not today, though 2.14 gets close:

| config | result |
| --- | --- |
| `aot_inductor.link_libtorch=False` | drops libtorch from `DT_NEEDED`, but leaves 23 `aoti_torch_*` symbols undefined for the host process to supply |
| `aot_inductor_mode.compile_standalone=True` | emits C++ source + `CMakeLists.txt` + `weights.o`, not a binary; its own link line is `-lgomp` only |
| built from that source by hand | `.so` with **no libtorch in `DT_NEEDED`** — but still 15 undefined `aoti_torch_*`, including `aoti_torch_cpu_addmm_out` |

Those shim symbols are exported by exactly one thing: `libtorch_cpu.so`, 421 MB.
No small shim library is shipped, and `find_package(TorchStandalone)` in the
generated CMake finds nothing in the wheel. So "standalone" currently means *no
link-time dependency*, not *no runtime dependency*: you would have to implement
the shim yourself, and `addmm` means bringing your own BLAS. The pip wheel also
ships no `.a` archives, so static linking would require building libtorch from
source.

The workable version of "bundle it" is what torch itself does: vendor
`libtorch_cpu.so` beside the artifact and set `RPATH=$ORIGIN` (torch's own libs
use exactly that). That is ~421 MB, or ~978 MB for all of `torch/lib`.

### Looking ahead: this may become the right answer

**Revisit AOTInductor periodically.** The blocker is not conceptual — it is that
the standalone path is unfinished in torch 2.14. The pieces are visibly being
built: `aot_inductor_mode.compile_standalone` and `aot_inductor.link_libtorch`
already exist and already remove libtorch from the link line, the generated code
already compiles against a `STANDALONE_TORCH_HEADER` shim, and the emitted
CMake already asks for a `TorchStandalone` package. What is missing is the other
half of that contract: the wheel ships no `TorchStandalone` CMake config, no
static archives, and no small runtime providing the ~15 `aoti_torch_*` shim
symbols — today only the 421 MB `libtorch_cpu.so` exports them.

If a future PyTorch ships that shim runtime, AOTInductor becomes the best option
on this list by a wide margin: a genuinely self-contained model with no libtorch,
no compile at startup, no Python-version coupling, and none of the cache-blob's
version fragility. It is worth re-testing on each release. The check is cheap and
already written:

```bash
python crimm_integration/examples/aoti_portability_probe.py build   # in the new env
python crimm_integration/examples/aoti_portability_probe.py load    # in an older one
```

Two things to look for on a new release: whether `find_package(TorchStandalone)`
resolves against something the wheel actually ships, and whether a
`compile_standalone` build still leaves `aoti_torch_*` symbols undefined
(`nm -D --undefined-only`). If both clear, the standalone artifact is real and
this document's recommendation should change.

---

## 4. ONNX — standalone, no PyTorch

### Export

```bash
srun --partition=cpu --cpus-per-task=16 --time=03:00:00 bash -lc '
  source $(conda info --base)/etc/profile.d/conda.sh && conda activate crimm-export
  cd /home/ziqiaoxu/RFdiffusion
  python tools/export/export_onnx.py --out models/rf_loop_dynamic.onnx'
```

Expect **about 35 minutes** on a dedicated 16-core node (the latest export
took 2101 s). Earlier exports of a larger graph took ~92 minutes on the busy
login node. The export produces **two** files, which must stay together in one
directory:

- `rf_loop_dynamic.onnx`: the graph (~36 MB)
- `rf_loop_dynamic.onnx.data`: the weights (~239 MB)

Then run the graph edits for ONNX Runtime (see *ONNX Runtime graph edits*
below). They take under a minute and write `rf_loop_dynamic_ort.onnx` plus its
own `.data` file:

```bash
python tools/export/onnx_ort_rewrites.py \
    models/rf_loop_dynamic.onnx models/rf_loop_dynamic_ort.onnx
```

In this checkout, **`models/rf_loop_dynamic_tp_ort.onnx` is the current
model**: the export `rf_loop_dynamic_tp.onnx` after both graph edits. The older
files are kept for comparison, and all give the same answers:

- `rf_loop_dynamic_mm*.onnx`: exported before the Transpose changes.
  `_mm_skipln` has the LayerNorm edit only; `_mm_ort` has both edits.
- `rf_loop_dynamic.onnx`: exported before the matmul rewrite.

The dynamo exporter writes the weights to a separate file to stay under
protobuf's 2 GB message limit. ONNX Runtime finds that file by the name
recorded in the graph.

**The sequence length `L` is symbolic by default**, so one file serves every
chain length. The relevant options are:

| flag | default | effect |
|---|---|---|
| `--lmin` / `--lmax` | 8 / 2048 | bounds declared for `L` |
| `--static` | off | bakes the traced length (L=76) into the graph, as the old export did |
| `--pdb`, `--contig` | 2KL8, `A1-20/6/A30-79` | the example input traced during export |

The bounds only guide the exporter. ONNX records the name of the dimension but
not its range, and ONNX Runtime does not check it. A model exported with
`--lmin 64` still runs correctly at L=27. Memory, not `--lmax`, is what limits
`L` in practice (see *Known limitations*).

### What made `L` dynamic, and how to keep it that way

A symbolic `L` survives export only if no code turns a shape into a plain
Python number. This model did that in one place: `key / math.sqrt(L)` in
`BiasedAxialAttention`. `math.sqrt` converts `L` to a float, which pins it to
the traced value, and every other `L` in the graph is pinned along with it.
Running the export with `TORCH_LOGS="+dynamic"` names the source line that
caused the first such pin.

A replacement must pass two separate stages, and the obvious one passes only
the first:

| spelling | `torch.export` | ONNX conversion |
|---|---|---|
| `math.sqrt(L)` | fails: `L` pinned to 76 | — |
| `torch.sym_sqrt(L)` | ok | fails: `No ONNX function found for torch._sym_sqrt` |
| `torch.sqrt(torch.scalar_tensor(L, ...))` | ok | ok, bit-identical |

The model uses the last form. **Test any new idiom like this on a four-line
module first.** Testing it on the real model means a 45-minute export that
fails at its final step.

### Why the attention uses matmul, not einsum

ONNX Runtime's CPU `Einsum` kernel is several times slower than PyTorch's. In
the first profile of the dynamic export, `Einsum` took **40%** of ONNX
Runtime's time. So the two most expensive contractions in
`BiasedAxialAttention` are written out as permute + matmul. The ~540 `Einsum`
nodes that remain are cheap and were left alone. Per forward pass at L=76:

| | with einsum | with matmul |
|---|---|---|
| `Einsum` nodes, ms | 1925 | 245 |
| `Transpose` nodes, ms (the rewrite's permutes) | 373 | 593 |
| ONNX Runtime total, ms | 4769 | **3307** |
| PyTorch eager total, ms | ~1660 | ~1660 |

This rewrite involves two trade-offs:

- **It is not bit-exact.** End to end, it moves `xyz_out` by 1.1e-5 Å and
  `alpha` by 1.1e-5 relative to the einsum version.
- **It uses more memory.** The faster form creates a temporary
  `(B, L, heads, L, L)` tensor on every attention call. That is ~7 MB at L=76,
  128 MB at L=200 and 432 MB at L=300.

This rewrite works against the tracing work described in MINIMAL_MODEL.md.
Replacing `opt_einsum.contract` with `torch.einsum` let dynamo trace the model,
but it is also what put 693 `Einsum` nodes in the ONNX graph. **Do not turn
these two contractions back into einsum calls.**

### Keeping Transpose cheap

After the LayerNorm edit (below), `Transpose` was the largest remaining cost:
355 ms of a 1618 ms forward pass at L=76 with 8 threads. ONNX Runtime's
`Transpose` copies its input on a **single thread**, so more threads do not
help, and the time depends on the permutation, not just on the tensor size. On
the same `[1,76,76,4,32]` tensor:

| permutation | what moves | ms per call |
|---|---|---|
| `(0,1,3,2,4)` | the 4-long axis moves inward; 32-element blocks are copied whole | 0.08 |
| `(0,3,2,1,4)` | general permutation; 32-element blocks | 0.49 |
| `(0,1,3,4,2)` | a 76-long axis moves to the end; one element at a time | 0.92 |

The exporter records each node's PyTorch source location, so the time can be
traced to model code. Two places accounted for nearly all of it:

1. **`se3/basis.py`, `update_basis_with_fused`: 205 ms.** It filled
   `torch.zeros` tensors by slice assignment. The exporter lowers each
   assignment to `ScatterND` wrapped in edge-sized `Transpose` nodes, about
   1300 per forward pass. These nodes show no source line in the export, only
   `<eval_with_key>`. The function now builds the same tensors with `F.pad`
   and `torch.cat`, which is bit-identical and exports with no `Transpose` or
   `ScatterND`.
2. **`BiasedAxialAttention`: 149 ms**, from the permutes of the matmul
   rewrite above. The attention now uses only cheap permutations, runs the
   softmax on the last axis (ONNX Runtime transposes internally for any other
   axis), and treats one of the two `L` axes as a batch axis for the value
   step. The key still needs its last two axes swapped. The exporter merges
   that swap back into the slow `(0,1,3,4,2)` permute, so the graph edit
   below turns it into a copy-free `FusedMatMul`. Against the old layout, the
   attention core runs 2x faster in PyTorch and 3x faster in ONNX Runtime
   with the edit, and the full model's outputs move by at most 2.9e-5 Å on
   `xyz_out` and 1.4e-5 on `alpha`.

On one real model block with 8 threads and both graph edits applied, ONNX
Runtime went from 45.3 ms to 32.8 ms (PyTorch: 23 ms), and `Transpose` from
9.4 ms to 0.7 ms. Across the whole model, `Transpose` fell from 355 ms to
26 ms, and a forward pass at L=76 from 1.18 s to 0.99 s, with both graph edits
applied before and after.

**When you edit the model, avoid two patterns:** slice assignment into a
preallocated tensor (`x[:, a:b] = y`), and permutes that move a long axis to
the end. `profile_onnx.py` shows whether either has crept back in.

### ONNX Runtime graph edits

`tools/export/onnx_ort_rewrites.py` edits an exported file in under a minute, with
no re-export. It makes three edits. The first two replace standard ONNX
operators with Microsoft-extension operators that ONNX Runtime runs faster.
The third works around an ONNX Runtime GPU bug using a standard operator.
Every other node and all the weights stay as they were.

- **With the first two edits, only ONNX Runtime can load the result.** The
  standard `onnxruntime` and `onnxruntime-gpu` packages include these
  operators; other ONNX runtimes do not. With `--no-layernorm
  --no-fused-matmul`, only the third edit is made and the file stays standard
  ONNX.
- Pass `--no-layernorm`, `--no-fused-matmul` or `--no-reducel2-fix` to skip an
  edit.
- The script writes to temporary files and renames them into place, so it is
  safe to regenerate a model that a running process has open.

#### LayerNorm -> SkipLayerNorm

ONNX Runtime's CPU kernel for the standard `LayerNormalization` op is slow,
even on one thread: 7.2 ms against PyTorch's 0.96 ms on a `[1,76,76,128]`
input. Its cost grows with the number of rows, and this model normalises
`L×L`-row pair features and `E`-row edge features about 600 times per forward
pass. ONNX Runtime's `SkipLayerNormalization` kernel, a Microsoft extension
that computes `LayerNorm(input + skip)`, is 2-5x faster on the same data.

The script rewrites all 1077 LayerNorm nodes:

```
LayerNormalization(x, scale, bias, epsilon=eps)
  ->  SkipLayerNormalization(x, x, scale, bias, epsilon=4*eps)
```

The two are the same computation. LayerNorm ignores the scale of its input
except through `eps`, and `x + x` is exact in floating point, so normalising
`2x` with `4*eps` gives exactly the result of normalising `x` with `eps`. The
kernel accepts only 2-D or 3-D inputs, so 4-D inputs are reshaped to 3-D and
back, which copies no data.

The effect, per forward pass at L=76 with 8 threads: LayerNorm 323 ms → 85 ms,
and the whole model 15-20% faster.

**Tried and rejected:** writing LayerNorm out as basic ops (`ReduceMean`,
`Sub`, `Mul`, ...). That was faster for a single LayerNorm, but it made a real
model block slower (46 ms → 52 ms).

#### Transpose before MatMul -> FusedMatMul

ONNX Runtime's `Transpose` copies its input on a single thread, and how long
that takes depends on the permutation. A permutation that moves the last axis
into second-to-last place, such as `(0,1,3,4,2)`, copies one element at a
time. When such a `Transpose` feeds only `MatMul` nodes, the script swaps its
last two output axes back and turns each `MatMul` into
`FusedMatMul(transA or transB = 1)`, which lets the matrix-multiply kernel read
the swapped axes without a copy. The `Transpose` that remains keeps the last
axis in place and copies whole blocks; if it has become the identity, it is
removed. The output is bit-identical.

The script also removes the shape recorded in the graph for each changed
`Transpose` output. ONNX Runtime plans memory from those recorded shapes, and
a stale one fails at run time with `Shape mismatch attempting to re-use
buffer`.

#### ReduceL2 over length-1 axes -> Abs (GPU correctness)

On ONNX Runtime's CUDA provider (checked with `onnxruntime-gpu` 1.23.2), the
model returned garbage coordinates of order 1e5, while the CPU provider was
correct. The cause is `ReduceL2`: on CUDA, when every reduced axis has length
1, it returns its input unchanged, sign included, instead of `|x|`. The SE(3)
layers take exactly that norm of their degree-0 features (`NormSE3`). A
negative "norm" is clamped to 2^-24, and dividing by it blows the features up
by ~1e7. A single-operator reproduction:

| input | reduced axis | CUDA result |
|---|---|---|
| `(76, 32, 1)` | last (length 1) | wrong: `[0.377, -0.396, 1.92]` for `[0.377, 0.396, 1.92]` |
| `(76, 32, 3)` | last (length 3) | correct |

The script turns each `ReduceL2` whose reduced axes all have length 1 into
`Abs`, which is the same value (40 nodes in this model). Current model code no
longer produces the pattern: `se3/layers.py:_feature_norm` uses `abs()` for
that case, which is bit-identical to `.norm()` in PyTorch. So only files
exported before that change need the edit, including `rf_loop_dynamic_tp.onnx`
in this checkout.

### Verify it — do not skip this

```bash
python tools/export/verify_onnx_model.py models/rf_loop_dynamic_ort.onnx --threads 16
```

**An export that reports SUCCESS is not necessarily correct.** The deprecated
TorchScript exporter produced a file that loaded cleanly, ran, and returned
values with relative error of order 1.0. They were completely wrong, and no
error was raised. The verifier does three things. It compares against PyTorch
on identical inputs. It repeats the comparison with jittered coordinates,
which changes the internal edge count `E` and would expose a graph that baked
`E` in as a constant. Only then does it time the model. Look for
`VERDICT: USABLE`. Run it on the file you will ship, including after the
LayerNorm pass.

The verifier checks one length. To check that one file really serves several
lengths, run:

```bash
python tools/export/bench_onnx.py models/rf_loop_dynamic_ort.onnx --threads 8
```

It runs L = 76, 80, 100 and 57 through a single session. It compares each
length against PyTorch and reports load time, first-call time and
steady-state time separately.

**Long chains cannot be checked this tightly, and the reason is the model, not
the export.** At L=270, the ONNX-vs-PyTorch difference in `alpha` exceeds the
1e-2 gate. However, nudging the input coordinates by a relative 1e-7 moves
PyTorch's *own* output by 5e-2 in `alpha` and 5.8e-2 Å in `xyz_out`, and the
nudge changes none of the top-k edges. At that length, the model amplifies
float32 rounding through its 40 blocks. At L=76, the same nudge moves the
outputs by only 1e-5 to 1e-3. For long chains, compare the resulting
structures, not the maximum absolute difference.

### Finding where the time goes

```bash
python tools/export/profile_onnx.py models/rf_loop_dynamic_ort.onnx --threads 8
```

This script runs ONNX Runtime's per-node profiler and `torch.profiler` on the
same inputs. It groups the time by operator type and prints the two results
side by side. Run it before you guess: two of the three causes predicted
before the first profile were wrong.

### Using the exported model

```python
import numpy as np, onnxruntime as ort

so = ort.SessionOptions()
so.intra_op_num_threads = 16
sess = ort.InferenceSession("models/rf_loop_dynamic_ort.onnx", so,
                            providers=["CPUExecutionProvider"])   # 4-26 s

outputs = sess.run(None, {
    "msa_latent": msa_latent, "msa_full": msa_full, "seq": seq, "xyz": xyz,
    "idx": idx, "t1d": t1d, "t2d": t2d, "xyz_t": xyz_t, "alpha_t": alpha_t,
})   # -> msa0, pair, xyz_out, state, alpha, logits_aa, pred_lddt  (all np.float32)
```

All inputs are `numpy` arrays, and every input has `L` as a symbolic axis.
Build them exactly as `MinimalLoopSampler._preprocess` does. The sampler still
does the pre- and post-processing; only the network itself is in the `.onnx`
file.

Usually, load the model and let the sampler do all of this:

```python
from crimm_ml.models import load_model
from crimm_ml.minimal_loop_sampler import MinimalLoopSampler

model = load_model("models/rf_loop_dynamic_ort.onnx", threads=8)   # device="cuda" for a GPU
sampler = MinimalLoopSampler(pdb_lines, contig, model, T=50)
results = sampler.sample(num_designs=4)          # progress bar when interactive
```

**Progress.** `sample(progress=...)` reports the reverse-diffusion steps, one
bar across all designs. By default it shows a tqdm bar when tqdm is installed
and the output is interactive (a terminal or a notebook), and prints nothing
otherwise, so batch logs stay clean. `progress=True` always reports (a bar, or
a line every 10 s when tqdm is missing), `False` never does, and it also takes
a callback `f(done, total, info)` or a bar of your own to update. tqdm stays
optional; `src/crimm_ml/progress.py` has the details.

**With an ONNX model, nothing imports PyTorch.** `load_model` picks the
backend from the file suffix (`.onnx`, or `.pt` for a checkpoint), and both
kinds are called the same way, with NumPy arrays. The sampler itself is NumPy
(`src/crimm_ml/numpy_rf`, ported from the torch code in `minimal_rf/`),
so the ONNX path needs only `numpy`, `scipy`, `omegaconf` and `onnxruntime`.
`tools/export/verify_numpy_sampler.py` checks the port against the torch pipeline
and runs the ONNX sampler with `import torch` blocked. A loaded model can be
passed to any number of samplers; `fill_gap_minimal` and
`MinimalPipelineLoopBuilder` take it the same way.

- **The checkpoint settings travel inside the ONNX file**, as metadata
  (`crimm_rf.config_dict`: the checkpoint's `model`, `diffuser` and
  `preprocess` sections). `export_onnx.py` writes them; for files exported
  before that, run `tools/export/onnx_embed_config.py <model.onnx>` once, on a
  machine that has torch and the checkpoint.
- **The diffusion schedule tables are cached as `schedules/*.npz`.** Upstream
  caches them as pickles that contain a PyTorch tensor, so they cannot be read
  without PyTorch. The sampler converts an existing pickle to `.npz` the first
  time it runs somewhere PyTorch is installed (this checkout's T=10, 15 and 50
  are already converted). With neither file, it computes the tables in NumPy,
  which takes about a minute. Those agree with upstream's wherever the
  distribution has meaningful density, and a T=50 run gave identical
  coordinates with either.
- **`threads=None`** keeps ONNX Runtime's default of one thread per physical
  core. Set it explicitly on a shared machine. `threads` applies to ONNX
  models only.
- **Same seed, different loop.** Upstream draws the translation noise from
  torch and the NumPy sampler from NumPy, so a seed does not reproduce an
  upstream trajectory. With the noise made identical, the two agree to float
  rounding (see the verify script).

**On a GPU**, install `onnxruntime-gpu`; `load_model` picks CUDA when the
CUDA provider is available (or pass `device="cuda"`). Three things to know:

- **Install only `onnxruntime-gpu`.** It and the CPU-only `onnxruntime`
  package install into the same directory, and having both installed leaves
  whichever was installed last. The `RFDiffusion` env currently has both.
- **ONNX Runtime does not raise if CUDA fails to load.** It logs an error
  (for example `libcublasLt.so.12: cannot open shared object file`) and runs
  on CPU. The CUDA 12 libraries in this env come from pip `nvidia-*` packages
  that are not on the loader path; the model calls
  `onnxruntime.preload_dlls()` to load them, then checks which provider is
  actually in use, warns if it is not CUDA, and reports it in `model.device`.
- **Use a model with the `ReduceL2` edit** (above), or it returns garbage.

Measured once on an RTX 2080 Ti (`gpu2080`, ONNX Runtime 1.23.2), sampling at
L=76, T=50: ONNX on CUDA took 0.16 s per step against 0.35 s for PyTorch eager
on the same GPU. The final coordinates agreed to 6.2e-6 Å.

### Known limitations

1. **It is about as fast as PyTorch eager on CPU, not faster.** These numbers
   come from `bench_onnx.py` with 8 threads on a dedicated 16-core node
   (`gollum006`, Xeon Gold 6336Y), using onnxruntime 1.29 and torch 2.14:

   | L | 57 | 76 | 80 | 100 |
   |---|---|---|---|---|
   | **current model** (`_tp_ort`) | **0.58 s** | **0.99 s** | **1.05 s** | **1.65 s** |
   | before the Transpose changes, both edits (`_mm_ort`) | 0.69 s | 1.18 s | 1.32 s | 2.14 s |
   | before the Transpose changes, LayerNorm edit only | 0.77 s | 1.25 s | 1.36 s | 2.16 s |
   | before the Transpose changes, no edits | 0.91 s | 1.56 s | 1.67 s | 2.62 s |
   | PyTorch eager | 0.61 s | 0.90 s | 0.95 s | 1.57 s |
   | current / PyTorch | 0.95x | 1.10x | 1.11x | 1.05x |

   `bench_onnx.py` times PyTorch with a single call, and those timings varied by
   up to ~10% between runs. At L=76, one design at T=50 therefore takes ~50 s
   of forward passes, against ~45 s in PyTorch. Loading the model takes 4-7 s
   when the 275 MB of files are already in the operating system's file cache,
   and up to 26 s when they are not.

   **Benchmark on a dedicated node.** On the shared login node `gollum`, which
   was busy with other work, both runtimes ran about 1.8x slower, and ONNX
   Runtime was hit harder: an earlier model took 3.12 s there at L=76, against
   PyTorch's 1.78 s. Do not compare numbers across the two machines.

   More threads barely help: the current model takes 0.96 s at 16 threads (in
   `verify_onnx_model.py`) against 0.99 s at 8. `bench_onnx.py`'s PyTorch
   timings are unreliable at 16 threads, because both runtimes' thread pools
   compete for the same 16 cores in one process.

   Where the remaining time goes, per forward pass at L=76 with 8 threads
   (`profile_onnx.py`, current model; the profiler adds some overhead to both
   totals):

   | op type | ONNX Runtime | share | PyTorch |
   |---|---|---|---|
   | `MatMul` + `FusedMatMul` + `Gemm` | 445 ms | 37% | `addmm` + `bmm` + `mm`, 361 ms |
   | `Einsum` | 130 ms | 11% | — |
   | `Pad` | 114 ms | 9% | — |
   | `ScatterElements` | 88 ms | 7% | `scatter_add_`, 12 ms |
   | `SkipLayerNormalization` | 80 ms | 7% | `native_layer_norm`, 46 ms |
   | `Transpose` | 26 ms | 2% | — |
   | total | 1207 ms | | 928 ms |

   The matrix multiplies are real work, running about 25% slower than
   PyTorch's. `Pad` comes from the new basis construction in `se3/basis.py`;
   it costs less than the slice assignments it replaced but is the next
   candidate. `Einsum` is the ~540 small contractions still written as einsum.
   `ScatterElements` is the SE(3) graph aggregation.

   The following were measured and ruled out as causes:
   - **Dynamic shapes.** The hot contractions ran at the same speed exported
     static or dynamic (1.00x), and shape arithmetic is ~2.4% of runtime.
   - **Graph aggregation.** The scatter/gather nodes take ~7.5% of runtime.
   - **Graph optimisation level.** `ORT_ENABLE_ALL`, `ORT_ENABLE_EXTENDED` and
     `ORT_DISABLE_ALL` are within 1% of each other. This was measured on the
     earlier fixed-`L` export.

2. **Memory grows as L³.** The matmul form of the attention creates a
   `(B, L, heads, L, L)` tensor on each call (432 MB at L=300). `--lmax` does
   not protect against this, because ONNX Runtime ignores it.
3. **Tight numeric checks stop working at long `L`.** See *Verify it* above.

---

## 5. Benchmarks

All on one node so the comparison is fair. L=76, single forward pass, median of 3.

**GPU vs CPU** (`gollum158`, RTX 2080 Ti + 16 cores):

| configuration | ms / forward | one design @ T=50 |
|---|---|---|
| CPU, 1 thread | 4054 | 203 s |
| CPU, 8 threads | 1240 | 62 s |
| CPU, 16 threads | 998 | 50 s |
| GPU eager | 336 | 17 s |
| GPU + `torch.compile` | **206** | **10 s** |

**PyTorch vs ONNX Runtime on CPU** (`gollum006`, 16 cores allocated). This table
was measured on the earlier **fixed-`L` export, before the matmul rewrite**. For
the current model (0.95-1.11x PyTorch's time at 8 threads), see §4 *Known limitations*.
It has not been re-measured on this node.

| runtime | 4 threads | 16 threads | one design @ T=50 |
|---|---|---|---|
| PyTorch | 1216 ms | **710 ms** | **36 s** |
| ONNX Runtime | 2704 ms | 2209 ms | 110 s |
| | (2.2x slower) | (3.1x slower) | |

### Does this model need a GPU?

**No.** A 16-core CPU runs one design at T=50 in ~36 s, against ~17 s on an RTX
2080 Ti and ~10 s compiled. That is a **2-3x** gap, not the 20-50x that usually
forces GPU deployment — because this model is small (the trunk is 36 blocks at
L=76) and spends much of its time in ops that do not saturate a GPU.

For loop building, where a run is a handful of designs per gap, CPU is
comfortably adequate. Use a GPU when you have one; do not require one.

CPU saturates at ~16 threads; beyond that, extra cores buy nothing.

---

## 6. What is verified, and what is not

| claim | evidence |
|---|---|
| SE(3)-Transformer without DGL/e3nn is bit-exact | `verify_no_dgl.py se3` → `0.000e+00`, both SE3 configs |
| `Base_ckpt.pt` loads unchanged | `strict=True`, 44 tensors, 0 missing/unexpected |
| model matches stock RFdiffusion | `verify_no_dgl.py model` → coords within 2.4e-4 Å |
| branch elimination is exact | `verify_minimal_rf_module.py` → BIT-EXACT at T=15, T=50 |
| runs with no dgl/e3nn/opt_einsum/rfdiffusion | import hook blocking all four |
| `torch.compile` gives 2.18x | `compile_se3_probe.py`, one run, all configs |
| `torch.export` captures the whole model | succeeds with 0 graph breaks: 185 s with a fixed `L`, ~390 s with a dynamic `L` |
| ONNX is numerically correct | `verify_onnx_model.py` → `VERDICT: USABLE`; `bench_onnx.py` matches PyTorch at L=57, 76, 80, 100 |
| ONNX is deterministic run to run | yes, after the `scatter_add_` fix |
| ONNX is *faster* | **no — about the same**: 0.95-1.11x PyTorch eager's time at L=57-100 (8 threads, dedicated node, current model); 2-3x slower before this work |
| ONNX model is correct on GPU | after the `ReduceL2` edit: CUDA forward pass within 4.3e-6 of PyTorch; sampler at T=50 on CUDA within 6.2e-6 Å of the PyTorch sampler (RTX 2080 Ti, ORT 1.23.2). Before it: garbage (~1e5) |
| the Transpose changes keep the results | `se3/basis.py` change bit-exact; model code old vs new in PyTorch: `xyz_out` <= 2.9e-5 Å, `alpha` <= 1.4e-5 at L=57, 76, 100; the `FusedMatMul` edit bit-exact; current model `VERDICT: USABLE` |
| the LayerNorm pass changes only speed | `verify_onnx_model.py` → `VERDICT: USABLE`; `bench_onnx.py` matches PyTorch at L=57, 76, 80, 100; every other node is byte-identical |
| ONNX handles variable sequence length | **yes** — one export, matches PyTorch at L=27, 57, 76, 80, 100; L=270 runs, but it cannot be checked tightly (§4) |
| matmul attention matches the einsum version | end to end: `xyz_out` 1.1e-5 Å, `alpha` 1.1e-5 |

### Four bugs found by exporting, three of them in stock RFdiffusion code

Exporting forces every operation to be stated precisely enough for a second
implementation to reproduce it, which surfaced problems that were always there:

1. **`index_add(dim=1, idx=[0])` in `MSAPairStr2MSA`** exports wrong whenever
   `N == 1` (relative error 0.75) — with shape `(B, 1, L, D)` two dimensions are
   size 1 and the lowering picks the wrong axis. Correct for `N >= 2`, which is
   why it went unnoticed; this pipeline always uses a single sequence, so it
   always hit the broken case.
2. **`torch.cdist` is inaccurate in eager**, not just in ONNX. Above 25 points
   it uses a matmul expansion, and the diagonal — a point's distance to itself,
   exactly zero — comes out at **1.1e-2 Å**. `compute_mode='donot_use_mm_for_euclid_dist'`
   fixes eager but fails to export at all. All four call sites now compute
   differences explicitly, which is bit-exact both ways.
3. **`torch.cross` without an explicit `dim`** exports incorrectly on shapes
   with more than one size-3 dimension (eager takes the first, ONNX the last).
   This model already passes `dim=-1` everywhere it matters, but it is a live
   hazard elsewhere in the codebase.
4. **A race in ONNX Runtime's scatter-with-add** — this one is ORT's bug, not
   RFdiffusion's. ORT parallelises `index_add_` over the inner dimension and
   races on duplicate destination indices, which every graph aggregation has by
   construction. At inner width <= 32 it is correct; at >= 64 it is wrong by
   15-28% **and unstable run to run** — three runs of one session gave three
   answers. Within a single SE(3) call the type-1 output (width 6) was
   bit-exact while type-0 (width 64) was garbage, which is what made it hard to
   find. `scatter_add_` is bit-identical in eager and exact through ORT, so
   `se3/graph.py` uses that.

All four were **silent wrong-answer bugs, not crashes**. None raised an error;
every one was caught only by comparing numbers against the reference.
