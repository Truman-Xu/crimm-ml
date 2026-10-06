"""Saving and loading the compiled model as a portable file.

This is the distribution mechanism `crimm` ships with. A compile of
`MinimalRoseTTAFoldModule` costs ~200 s; `torch.compiler.save_cache_artifacts()`
bundles the result into a single file that can travel inside the wheel, and
loading it takes the first forward pass down to ~3.5 s. Nothing is compiled on
the user's machine, no compiler is invoked, and no GPU is involved.

Measured on CPU with the on-disk inductor cache deliberately **wiped**, so only
the blob could account for the difference. "Setup" is everything a fresh process
pays before its first real forward -- loading the blob, wrapping with
`torch.compile`, and one warmup pass:

    configuration                blob    torch.compile()   warmup     setup
    no blob (control)              --            1.17 s   204.4 s   205.5 s
    blob + caching_precompile  25.1 MB          27.71 s     4.0 s    31.9 s

**Time the whole of that, not just the warmup.** With `caching_precompile` the
work moves into the `torch.compile()` call itself rather than the first forward,
so timing only the warmup reports ~3.5 s and hides ~28 s. The honest number is
a **6.4x** reduction in startup, not the 55x that slice suggests.

That is still decisive against a 205 s compile, and it is why
`compilation.configure()` sets `caching_precompile` *before* anything compiles:
without it the blob carries inductor's half and dynamo re-traces from scratch in
every process. The loaded model stays shape-dynamic either way -- a second
sequence length costs well under a second, not another compile.

Break-even, at L=76 on 16 CPU threads (eager 0.79 s/forward, compiled 0.42 s):
one design at T=50 is 50 forwards, so eager costs ~40 s against ~53 s for
blob-compiled. **Compiling wins from about the second design onward** and grows
from there. It is not free at one design, and this module does not pretend
otherwise.

It survives a different machine. Saved on an Ice Lake node and loaded on a
Cascade Lake one, the warmup was 8.4 s rather than 3.5 s -- the blob still took
effect on hardware it had never seen. (Those two are warmup-only figures, taken
before the accounting above; the cross-machine total was not re-measured.)

**A blob is keyed to a torch version and is not hardware-neutral.** Treat a miss
as normal: `load()` returns False rather than raising, and the caller compiles
as usual. Never make a missing or stale blob fatal -- that is the property that
made this preferable to AOTInductor, whose artifacts fail at `dlopen` instead.
See `DISTRIBUTING.md`.

Producer side (once, offline, per torch version you support):

    from crimm_ml import compiled_model
    compiled_model.build(model, lengths=(76,))     # compiles, then saves

Consumer side (in the installed package):

    compiled_model.load()                          # before building the model
    compilation.compile_model(model)               # now nearly free
"""
import time
from pathlib import Path

import torch

from . import compilation

#: Where a shipped blob lives inside the installed package. `build()` writes
#: here by default so the file lands somewhere `pip` will actually include.
DEFAULT_BLOB_PATH = Path(__file__).resolve().parent / "data" / "rf_loop_compile_cache.bin"


def save(path=DEFAULT_BLOB_PATH, verbose=True):
    """Write everything compiled so far in this process to `path`.

    Call *after* compiling and running at least one forward pass -- there is
    nothing to save before the compile has happened. Returns the path, or
    `None` if torch recorded no artifacts (which means nothing was compiled, or
    it all came from a cache and was not re-recorded).
    """
    compilation.require_supported_torch()
    out = torch.compiler.save_cache_artifacts()
    if out is None:
        if verbose:
            print("[compiled_model] nothing to save -- compile something first",
                  flush=True)
        return None
    blob, info = out
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    if verbose:
        kinds = {k: len(v) for k, v in getattr(info, "artifacts", {}).items()}
        print(f"[compiled_model] saved {len(blob) / 1e6:.1f} MB -> {path}  {kinds}",
              flush=True)
    return path


def load(path=DEFAULT_BLOB_PATH, verbose=True):
    """Load a blob so the next compile is nearly free. Returns True on success.

    Never raises on a missing, unreadable or mismatched blob -- a miss just
    means the model compiles normally, which is correct behaviour rather than an
    error. Call this **before** compiling.

    Sets `caching_precompile` itself rather than trusting the caller to have
    configured torch first. Order matters more than it looks: loading a blob
    before that flag is set consumes only inductor's artifacts and leaves
    dynamo to re-trace, which measured **28.8 s of setup instead of 3.5 s** in
    the end-to-end test. Making the flag this function's own precondition means
    no call order can quietly cost that.
    """
    path = Path(path)
    if not path.exists():
        if verbose:
            print(f"[compiled_model] no blob at {path}; will compile normally",
                  flush=True)
        return False
    ok, reason = compilation.blob_support()
    if not ok:
        if verbose:
            print(f"[compiled_model] {reason}; ignoring blob", flush=True)
        return False
    import torch._dynamo.config as dcfg
    dcfg.caching_precompile = True
    try:
        t0 = time.time()
        torch.compiler.load_cache_artifacts(path.read_bytes())
    except Exception as exc:                      # noqa: BLE001 - see docstring
        if verbose:
            print(f"[compiled_model] blob at {path} unusable ({type(exc).__name__}: "
                  f"{exc}); will compile normally", flush=True)
        return False
    if verbose:
        print(f"[compiled_model] loaded {path.stat().st_size / 1e6:.1f} MB from "
              f"{path} in {time.time() - t0:.2f}s", flush=True)
    return True


def build(model, path=DEFAULT_BLOB_PATH, lengths=(76,), device=None,
          dynamic=True, verbose=True):
    """Compile `model` and write the result to `path`. The producer side.

    One length is enough with `dynamic=True`: the graph is shape-generic, so the
    `L` compiled here does not have to match anything a user will run. Pass more
    lengths only if you have a reason to believe a particular size takes a
    different path.

    Returns the blob path, or `None` if torch recorded nothing.
    """
    compilation.configure(verbose=verbose)
    ok, msg = compilation.toolchain_ok()
    if not ok:
        raise RuntimeError(f"cannot compile: {msg}")
    compilation.compile_model(model, dynamic=dynamic, configure_cache=False)
    for L in lengths:
        t0 = time.time()
        model.warmup(L, device=device)
        if verbose:
            print(f"[compiled_model] compiled at L={L} in {time.time() - t0:.1f}s",
                  flush=True)
    return save(path, verbose=verbose)


def load_and_compile(model, path=DEFAULT_BLOB_PATH, warmup_L=None, device=None,
                     dynamic=True, verbose=True):
    """The consumer side: load a blob if there is one, then compile.

    This is the call an application makes. With a matching blob the compile is
    a cache lookup; without one it is a normal compile. Either way the model
    comes back ready and shape-dynamic.

    `warmup_L` runs one throwaway forward to force the compile now rather than
    inside the caller's first real call. Any length will do.
    """
    compilation.configure(verbose=False)
    try:
        load(path, verbose=verbose)
        compilation.compile_model(model, dynamic=dynamic, configure_cache=False)
    except Exception as exc:                      # noqa: BLE001 - see below
        # KNOWN ISSUE (torch 2.14): replaying a `caching_precompile` blob can
        # abort inside dynamo with
        #     AssertionError: Name '__builtins_dict___3' already exists in scope
        # reproducibly, on an empty cache and a warm one alike. Compiling is an
        # optimisation, so a failure here must never take the caller's run down
        # with it -- fall back to the eager model and say so loudly.
        import torch._dynamo as dynamo
        print(f"[compiled_model] compile failed ({type(exc).__name__}: {exc}); "
              f"falling back to the eager model. See DISTRIBUTING.md, "
              f"'Known issue'.", flush=True)
        compilation.disable_compile(model)
        try:
            dynamo.reset()
        except Exception:
            pass
        return model
    if warmup_L is not None:
        t0 = time.time()
        model.warmup(warmup_L, device=device)
        if verbose:
            print(f"[compiled_model] ready in {time.time() - t0:.1f}s "
                  f"(first forward at L={warmup_L})", flush=True)
    return model


def info(path=DEFAULT_BLOB_PATH):
    """`dict` describing a blob on disk, without loading it into torch."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {"path": str(path), "exists": True,
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "torch_that_can_load_it": "the version it was built with"}
