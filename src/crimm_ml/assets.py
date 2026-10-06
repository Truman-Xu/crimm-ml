"""The model file: where it is cached, and how it gets there.

The 275 MB ONNX model is too large for a wheel (PyPI allows 100 MB per file),
so it is fetched once into a cache and checked against the sizes and SHA-256
checksums in `data/assets.json`. The cache lives at `$CRIMM_ML_CACHE` when set
(a container points it at a directory baked into the image), else at the
platform's user cache directory (`~/.cache/crimm-ml` on Linux).

    crimm-ml assets fetch                 # download (needs network)
    crimm-ml assets fetch --from DIR      # or copy from a directory you have
    crimm-ml assets verify                # full checksum of what is cached

The model is two files -- the graph and its `.onnx.data` weights -- and ONNX
Runtime finds the weights by the bare file name recorded inside the graph, so
both always live in the same directory under their original names.
"""
import hashlib
import json
import logging
import os
import time
import urllib.request
import uuid
from importlib import resources
from pathlib import Path

from platformdirs import user_cache_dir

from .errors import AssetError

__all__ = ["cache_dir", "schedule_dir", "model_dir", "model_path", "fetch", "verify", "manifest"]

log = logging.getLogger(__name__)
_CHUNK = 1 << 22


def cache_dir():
    """Root of the crimm_ml cache. Not created here: writers create what they need."""
    env = os.environ.get("CRIMM_ML_CACHE")
    return Path(env) if env else Path(user_cache_dir("crimm-ml", appauthor=False))


def schedule_dir():
    """The IGSO(3) schedule tables (`T_*.npz`), one ~8 MB file per T."""
    return cache_dir() / "schedules"


def model_dir():
    return cache_dir() / "models"


def manifest():
    text = resources.files("crimm_ml").joinpath("data").joinpath("assets.json").read_text()
    return json.loads(text)["model"]


def model_path(fetch_if_missing=False):
    """Path of the cached `.onnx` graph, its `.onnx.data` next to it.

    Checks presence and sizes only (instant); `verify()` does the checksums.
    Raises `AssetError` when the files are not there, unless
    `fetch_if_missing`, in which case they are downloaded first.
    """
    m = manifest()
    d = model_dir()
    problems = _size_problems(d, m["files"])
    if problems and fetch_if_missing:
        fetch()
        problems = _size_problems(d, m["files"])
    if problems:
        raise AssetError("The crimm_ml model is not in the cache (" + "; ".join(problems) +
                         "). Run `crimm-ml assets fetch`, or `crimm-ml assets fetch --from DIR` "
                         "with a directory that has the files.")
    return d / m["files"][0]["name"]


def verify(directory=None):
    """Check every model file's size and SHA-256. Returns a list of problems
    (empty when all is well)."""
    m = manifest()
    d = Path(directory) if directory else model_dir()
    problems = _size_problems(d, m["files"])
    if problems:
        return problems
    for f in m["files"]:
        got = _sha256(d / f["name"])
        if got != f["sha256"]:
            problems.append(f"{f['name']}: checksum {got[:12]}..., expected {f['sha256'][:12]}...")
    return problems


def fetch(source=None, force=False, retries=3):
    """Put the model files into the cache, checking each one's checksum.

    `source`: None downloads from the URLs in `assets.json`; a directory copies
    from it; any other string is a base URL that `/<file name>` is appended
    to. Files already present with the right checksum are kept unless
    `force`. Each file is written under a temporary name and renamed into
    place only once its checksum matches, so an interrupted fetch never
    leaves a file that looks complete.
    """
    m = manifest()
    dest = model_dir()
    dest.mkdir(parents=True, exist_ok=True)
    for f in m["files"]:
        target = dest / f["name"]
        if not force and target.exists() and target.stat().st_size == f["size"] \
                and _sha256(target) == f["sha256"]:
            log.info("%s: already cached", f["name"])
            continue
        candidates = _candidates(source, m, f["name"])
        if not candidates:
            raise AssetError("No download location is configured for the model yet. Copy it "
                             "from a directory that has it: `crimm-ml assets fetch --from DIR`.")
        if not any(_fetch_one(c, target, f, retries) for c in candidates):
            raise AssetError(f"could not obtain {f['name']} from {', '.join(candidates)}")
    return dest / m["files"][0]["name"]


def _candidates(source, m, name):
    if source is None:
        return [u.format(name=name) for u in m["urls"]]
    if os.path.isdir(source):
        return [os.path.join(source, name)]
    return [source.rstrip("/") + "/" + name]


def _fetch_one(src, target, f, retries):
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    attempts = 1 if os.path.exists(src) else retries
    for attempt in range(1, attempts + 1):
        try:
            log.info("%s <- %s", f["name"], src)
            digest = _copy_hashing(src, tmp)
            if digest != f["sha256"]:
                raise AssetError(f"{f['name']} from {src}: checksum mismatch")
            os.replace(tmp, target)
            return True
        except (OSError, AssetError) as e:
            log.warning("%s: attempt %d/%d failed: %s", f["name"], attempt, attempts, e)
            _remove(tmp)
            if attempt < attempts:
                time.sleep(2 ** attempt)
    return False


def _copy_hashing(src, dst):
    h = hashlib.sha256()
    opener = (lambda: open(src, "rb")) if os.path.exists(src) else \
             (lambda: urllib.request.urlopen(src, timeout=60))
    with opener() as fin, open(dst, "wb") as fout:
        while block := fin.read(_CHUNK):
            h.update(block)
            fout.write(block)
    return h.hexdigest()


def _size_problems(d, files):
    problems = []
    for f in files:
        p = d / f["name"]
        if not p.exists():
            problems.append(f"{p} is missing")
        elif p.stat().st_size != f["size"]:
            problems.append(f"{p} is {p.stat().st_size} bytes, expected {f['size']}")
    return problems


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(_CHUNK):
            h.update(block)
    return h.hexdigest()


def _remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
