"""Compiling `MinimalRoseTTAFoldModule`, and making the compile persist.

This module owns everything to do with `torch.compile` for this package. The
model itself (`minimal_rf_module.py`) deliberately knows nothing about it --
compiling is a deployment concern, not a property of the network.

Two layers of persistence, and they are different things:

- **The on-disk inductor cache**, configured by `configure()` below. It is
  machine-local, and it only ever helps the *next process on this machine*.
- **The portable cache blob**, in `compiled_model.py`. That is the one that
  ships inside `crimm`, and it is what makes a user's first forward pass cost
  seconds instead of minutes.

`configure()` exists mainly to move the inductor cache off `/tmp`. Inductor
defaults to `tempfile.gettempdir()/torchinductor_$USER`, and on a SLURM cluster
`/tmp` is node-local: a cache written by a job on one node is invisible to the
next job, which usually lands on a different one. Pointing it at `$HOME` (NFS,
visible fleet-wide) is the whole trick. Triton's own kernel cache already
defaults to `~/.triton/cache` on shared home and is left alone.

Cache safety across mixed hardware is enforced here, not assumed. Inductor's key
is supposed to cover the device, but a `$HOME` cache shared across a fleet was
observed handing a CPU-only job a CUDA entry, which fails with
`RuntimeError: No CUDA GPUs are available` from inside codegen. `configure()`
therefore appends a device tag (`cpu`, `cuda-sm75`, ...) to the cache root, so
CPU and each GPU architecture get physically separate directories. The
directories only grow; `clear_cache()` resets the one in use.

Usage:
    from crimm_ml import compilation
    compilation.configure()                 # before the first compile
    compilation.compile_model(model)        # dynamic shapes by default
"""
import os
import shutil
import tempfile
from pathlib import Path

#: Snapshot taken **before** any inductor import. This matters: importing
#: `torch._inductor.codecache` writes `TORCHINDUCTOR_CACHE_DIR` into
#: `os.environ` itself, defaulting to node-local `/tmp/torchinductor_$USER`.
#: Reading the variable after that point cannot distinguish a user's deliberate
#: choice from torch's own default, so the redirect this module exists to
#: perform would silently never happen.
_USER_CACHE_DIR_ENV = os.environ.get("TORCHINDUCTOR_CACHE_DIR")

import torch

#: Default cache root. `$HOME` rather than `/tmp` on purpose -- see module
#: docstring. Override with `configure(cache_dir=...)` or by exporting
#: `TORCHINDUCTOR_CACHE_DIR` yourself (an existing env var always wins).
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "crimm_rfdiffusion" / "torchinductor"

#: The release that introduced `torch.compiler.save_cache_artifacts`. The real
#: gate is the capability check in `require_supported_torch()` -- this is only
#: used to make the error message actionable.
MIN_TORCH_RELEASE = "2.7"

_state = {}


def blob_support():
    """`(ok, reason)` -- can this torch produce and consume a portable blob?

    Checked by capability rather than by version number, because the two pieces
    landed in different releases and a version comparison would be guessing
    about builds this was never tested against.
    """
    missing = [n for n in ("save_cache_artifacts", "load_cache_artifacts")
               if not hasattr(torch.compiler, n)]
    if missing:
        return False, (f"torch {torch.__version__} has no "
                       f"torch.compiler.{missing[0]}; the portable compile blob "
                       f"needs torch >= {MIN_TORCH_RELEASE}")
    import torch._dynamo.config as dcfg
    if not hasattr(dcfg, "caching_precompile"):
        return False, (f"torch {torch.__version__} has no "
                       "torch._dynamo.config.caching_precompile; without it a "
                       "blob carries only inductor's half of the compile and "
                       "dynamo re-traces in every process (measured: 114 s "
                       "instead of 3.5 s). Use a newer torch.")
    return True, f"torch {torch.__version__}"


def require_supported_torch():
    """Raise unless this torch supports the shipped-blob distribution model.

    This package targets exactly the torch versions that can load a portable
    compile blob. Older ones are not supported: they would silently fall back to
    a ~200 s compile in every process, which is the cost the whole distribution
    strategy exists to avoid. See `DISTRIBUTING.md`.
    """
    ok, reason = blob_support()
    if not ok:
        raise RuntimeError(reason)
    return reason


def _device_tag():
    """A cache-directory suffix that cannot collide across device types.

    Inductor's cache key is supposed to cover the device, but a `$HOME` cache
    shared fleet-wide has been observed handing a CPU-only job an entry whose
    generated module calls `async_compile.triton(...)` at import, which then
    fails with `RuntimeError: No CUDA GPUs are available` from deep inside
    codegen. Keeping CPU and each GPU architecture in separate directories
    makes that impossible rather than unlikely.
    """
    try:
        if torch.cuda.is_available():
            return "cuda-sm" + "".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception:                              # no driver, no CUDA build
        pass
    return "cpu"


def _torch_default_cache_dir():
    """What inductor would pick on its own: `<tempdir>/torchinductor_<user>`."""
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "nouser"
    return Path(tempfile.gettempdir()) / f"torchinductor_{user}"


def _resolve_cache_dir(explicit=None):
    """Explicit argument > a value the user really set > our shared default.

    A value matching torch's own default is treated as unset even if it is in
    the environment now, because inductor writes it there at import time.
    """
    if explicit is not None:
        return Path(explicit)
    env = _USER_CACHE_DIR_ENV or os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if env and Path(env) != _torch_default_cache_dir():
        return Path(env)
    return DEFAULT_CACHE_DIR / _device_tag()


def configure(cache_dir=None, move_triton_cache=False, fx_graph_cache=True,
              caching_precompile=True, verbose=True):
    """Point the inductor cache somewhere durable and enable blob capture.

    Idempotent, and safe to call before or after the model is built -- inductor
    reads `TORCHINDUCTOR_CACHE_DIR` lazily at the first compile, not at import.
    Returns the cache directory in use.

    `caching_precompile=True` is what lets a later `compiled_model.save()`
    capture dynamo's work as well as inductor's. It must be set **before**
    anything is compiled, which is why it lives here rather than in the saver,
    and it costs nothing on the compile itself (measured 201 s against 200 s).

    A `TORCHINDUCTOR_CACHE_DIR` the caller really set is never overridden, so a
    job script can still point the cache at node-local scratch. A value equal to
    torch's own default does not count as set -- see `_USER_CACHE_DIR_ENV`.
    """
    require_supported_torch()
    import torch._dynamo.config as dcfg
    import torch._inductor.config as icfg

    cache_dir = _resolve_cache_dir(cache_dir)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # On by default from torch 2.6; `False` is only useful for measuring what
    # the cache is worth -- see the RFdiffusion fork's
    # `crimm_integration/examples/compile_cache_probe.py --no-fx-cache`.
    icfg.fx_graph_cache = fx_graph_cache
    dcfg.caching_precompile = caching_precompile

    if move_triton_cache:
        # Off by default: triton's own default is already on shared home, so
        # moving it buys nothing except one directory to delete instead of two
        # -- at the cost of orphaning whatever is already cached there.
        os.environ.setdefault("TRITON_CACHE_DIR", str(cache_dir.parent / "triton"))

    # Cheap insurance. With dynamic shapes the model needs ~18 entries; the
    # default limit is comfortably above that, but a graph break pattern that
    # changes could push it over, and the failure mode is a silent fallback to
    # eager rather than an error.
    dcfg.cache_size_limit = max(dcfg.cache_size_limit, 64)

    _state["cache_dir"] = cache_dir
    if verbose:
        print(f"[compilation] TORCHINDUCTOR_CACHE_DIR={cache_dir} "
              f"(fx_graph_cache={icfg.fx_graph_cache}, "
              f"caching_precompile={dcfg.caching_precompile})", flush=True)
    return cache_dir


def compile_model(model, dynamic=True, configure_cache=True, **compile_kwargs):
    """`torch.compile` a model in place. Returns it, so it chains.

    Compiles `model.forward` and rebinds it on the instance rather than
    returning `torch.compile(model)`'s `OptimizedModule` wrapper, so a caller
    that already holds the object keeps working -- same class, same
    `state_dict`, same `warmup`.

    `dynamic=True` makes the sequence length symbolic, so **one compile serves
    every target length**. Measured over L = 76, 80, 100 and 57 in one process,
    it compiled 18 graphs where `dynamic=False` needed 57, and absorbed every
    length after the first with zero further graphs -- including L=57, below
    `make_topk_graph`'s `top_k=64` where `min(top_k, L)` looks like it must
    specialise and does not. It is also the faster and far more uniform of the
    two (0.30-0.67 s per forward against 0.46-5.55 s). See COMPILING.md.

    Note `torch._dynamo.mark_dynamic` on the input tensors is *not* a working
    substitute: the model graph-breaks at the `nonzero` fence in
    `util_module._edge_index_from_cond`, and every break hands dynamo fresh
    intermediates carrying concrete shapes. Marking only the entry tensors
    leaves those static and raises `ConstraintViolationError` at the
    `torch.cat` in `Track_module.py`. Only `dynamic=True` reaches them.

    The first forward pass after this call *is* the compile. Do it once, up
    front (`model.warmup(L)`), not inside a sampling loop -- and with
    `dynamic=True` that warmup covers every length, so the `L` you pass need not
    match the target you then run.
    """
    if is_compiled(model):
        return model
    if configure_cache:
        configure(verbose=False)
    else:
        require_supported_torch()
    model.forward = torch.compile(model.forward, dynamic=dynamic, **compile_kwargs)
    model._compiled = True
    return model


def disable_compile(model):
    """Undo `compile_model`, restoring the eager `forward`. For A/B measurement;
    the compiled artifacts stay in the cache.

    Deletes the instance attribute rather than reassigning it, so lookup falls
    back to the class's own `forward` and no bound method is left holding a
    reference to the model.
    """
    if is_compiled(model):
        del model.forward
        model._compiled = False
    return model


def is_compiled(model):
    """Whether `compile_model` has been applied to this instance."""
    return bool(getattr(model, "_compiled", False))


def cache_dir():
    """The directory `configure()` selected, or the one torch would use anyway."""
    if "cache_dir" in _state:
        return _state["cache_dir"]
    from torch._inductor.codecache import cache_dir as _inductor_cache_dir
    return Path(_inductor_cache_dir())


def cache_stats():
    """`(hits, misses, bytes_on_disk)` for the FX-graph cache so far.

    Hits/misses are dynamo's own counters for this process -- the honest way to
    tell a warm start from a cold one, rather than inferring it from wall time.
    """
    from torch._dynamo.utils import counters
    ind = counters.get("inductor", {})
    d = Path(cache_dir())
    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0
    return ind.get("fxgraph_cache_hit", 0), ind.get("fxgraph_cache_miss", 0), size


def clear_cache():
    """Delete the on-disk inductor cache. The next compile is cold again."""
    d = Path(cache_dir())
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def toolchain_ok():
    """`(ok, message)` -- whether the C++ compiler on PATH can build what
    TorchInductor emits.

    Inductor's generated C++ is C++20; this cluster's default g++ 8.5.0 cannot
    compile it and the failure surfaces deep inside codegen. `module load
    gcc/15.2.0` before python fixes it. Checked rather than assumed, because the
    failure mode is otherwise very confusing.

    Only matters when something actually compiles. A process that loads a blob
    covering everything it runs never invokes the compiler at all.
    """
    import subprocess
    try:
        out = subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout
        first = out.splitlines()[0]
    except (OSError, IndexError):
        return False, "no g++ on PATH; run `module load gcc/15.2.0`"
    major = 0
    for tok in first.replace(")", " ").split():
        if tok and tok[0].isdigit() and "." in tok:
            major = int(tok.split(".")[0])
            break
    if major < 9:
        return False, (f"g++ {major} is too old for TorchInductor's C++20 codegen "
                       f"({first}); run `module load gcc/15.2.0` before python")
    return True, first
