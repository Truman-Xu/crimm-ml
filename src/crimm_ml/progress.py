"""Progress reporting for the reverse-diffusion loop.

`MinimalLoopSampler.sample(progress=...)` passes its argument here. tqdm is
optional: it is used when installed, and there is a plain-text fallback, so
the ONNX path keeps its small dependency set (numpy, scipy, omegaconf,
onnxruntime).

Accepted values:

    None / "auto"   a bar when tqdm is installed *and* the output looks
                    interactive (a terminal, or a notebook kernel); nothing
                    otherwise, so batch logs stay clean.  (default)
    True            always report: a tqdm bar if tqdm is installed, else a
                    line on stderr every few seconds.
    False           nothing.
    a callable      called as `f(done, total, info)` after every step, where
                    `info` is `{"design": i, "num_designs": n, "t": t}` --
                    for a GUI, a log file, or a remote status.
    a tqdm-like     anything with `.update(n)`: used as given, and left open
    object          (the caller closes it), so several samplers can share one
                    bar.

Typical use:

    sampler.sample(num_designs=4)                  # bar when interactive
    sampler.sample(num_designs=4, progress=True)   # always
    sampler.sample(num_designs=4, progress=lambda done, total, info: ...)
"""
import sys
import time

__all__ = ["make"]


def make(progress, total, desc="reverse diffusion"):
    """A reporter with `.step(info)` and `.close()` for `total` steps."""
    # `is` and isinstance first: a tqdm object's `==` raises on a string.
    if progress is False:
        return _Null()
    if progress is True:
        return _tqdm(total, desc) or _Text(total, desc)
    if progress is None or (isinstance(progress, str) and progress == "auto"):
        return (_tqdm(total, desc) or _Null()) if _interactive() else _Null()
    if hasattr(progress, "update"):
        return _External(progress)
    if callable(progress):
        return _Callback(progress, total)
    raise TypeError(
        f"progress must be None, 'auto', True, False, a callable or a tqdm-like "
        f"object, got {type(progress).__name__}")


def _interactive():
    if "ipykernel" in sys.modules:   # a notebook kernel
        return True
    try:
        return bool(sys.stderr.isatty())
    except Exception:                # a stream that does not answer
        return False


def _tqdm(total, desc):
    """A tqdm bar, or None if tqdm is not installed."""
    try:
        from tqdm.auto import tqdm   # .auto gives a widget in notebooks
    except ImportError:
        return None
    return _Tqdm(tqdm(total=total, desc=desc, unit="step"))   # leave it: the final line is the record


class _Null:
    def step(self, info=None):
        pass

    def close(self):
        pass


class _Tqdm(_Null):
    def __init__(self, bar):
        self.bar = bar

    def step(self, info=None):
        if info and info.get("num_designs", 1) > 1:
            self.bar.set_postfix(design=f"{info['design']}/{info['num_designs']}", refresh=False)
        self.bar.update(1)

    def close(self):
        self.bar.close()


class _Text(_Null):
    """Fallback without tqdm: one line on stderr, at most every `every` seconds."""

    def __init__(self, total, desc, every=10.0):
        self.total, self.desc, self.every = total, desc, every
        self.done = 0
        self.t0 = self.last = time.monotonic()

    def step(self, info=None):
        self.done += 1
        now = time.monotonic()
        if now - self.last < self.every and self.done != self.total:
            return
        self.last = now
        per = (now - self.t0) / self.done
        eta = per * (self.total - self.done)
        print(f"{self.desc}: {self.done}/{self.total} steps "
              f"({100 * self.done / self.total:.0f}%), {per:.2f}s/step, eta {eta:.0f}s",
              file=sys.stderr, flush=True)

    def close(self):
        if self.done:
            print(f"{self.desc}: done, {self.done} steps in "
                  f"{time.monotonic() - self.t0:.1f}s", file=sys.stderr, flush=True)


class _Callback(_Null):
    def __init__(self, fn, total):
        self.fn, self.total, self.done = fn, total, 0

    def step(self, info=None):
        self.done += 1
        self.fn(self.done, self.total, info or {})


class _External(_Null):
    """A bar the caller owns: updated, never closed."""

    def __init__(self, bar):
        self.bar = bar

    def step(self, info=None):
        self.bar.update(1)
