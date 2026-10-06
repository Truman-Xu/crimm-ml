"""Fast checks that need no model file."""
import os
import stat
import subprocess
import sys
import textwrap

import numpy as np
import pytest


def test_import_does_not_need_torch_or_crimm():
    code = textwrap.dedent("""
        import importlib.abc, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("torch", "crimm"):
                    raise ImportError("blocked: " + name)
        sys.meta_path.insert(0, Block())
        import crimm_ml
        from crimm_ml import MinimalLoopSampler, load_model, ContigError
        from crimm_ml.numpy_rf import diffusion, denoise, geometry, igso3
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_contig_errors_are_exceptions_not_exits():
    from crimm_ml import ContigError, CrimmMLError
    from crimm_ml.minimal_rf.contigs import ContigMap

    with pytest.raises(ContigError) as e:
        ContigMap(parsed_pdb={"pdb_idx": []}, contigs=None, ref_idx=None)
    assert isinstance(e.value, ValueError) and isinstance(e.value, CrimmMLError)


def test_packaged_base_config():
    from crimm_ml.minimal_loop_sampler import _load_base_config

    conf = _load_base_config(None, None)
    for section in ("diffuser", "denoiser", "preprocess", "contigmap", "potentials", "model"):
        assert section in conf


def test_schedule_dir_precedence(tmp_path, monkeypatch):
    from crimm_ml import assets
    from crimm_ml.minimal_loop_sampler import _resolve_schedule_dir

    monkeypatch.setenv("CRIMM_ML_CACHE", str(tmp_path / "cache"))
    assert _resolve_schedule_dir(None, None) == str(tmp_path / "cache" / "schedules")
    assert assets.schedule_dir() == tmp_path / "cache" / "schedules"
    assert _resolve_schedule_dir(None, "/rf") == os.path.join("/rf", "schedules")
    assert _resolve_schedule_dir(tmp_path / "x", "/rf") == str(tmp_path / "x")
    assert not (tmp_path / "cache").exists()      # resolving never creates anything


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores permissions")
def test_schedule_cache_tolerates_read_only_dir(tmp_path, caplog):
    from crimm_ml.numpy_rf.diffusion import _write_npz_atomic

    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        _write_npz_atomic(str(ro / "t.npz"), {"a": np.arange(3)})   # must not raise
    finally:
        ro.chmod(stat.S_IRWXU)
    assert "Could not cache" in caplog.text
    assert list(ro.iterdir()) == []                 # no temp file left behind


def test_schedule_cache_write_is_complete(tmp_path):
    from crimm_ml.numpy_rf.diffusion import _write_npz_atomic

    _write_npz_atomic(str(tmp_path / "s" / "t.npz"), {"a": np.arange(5)})
    assert [p.name for p in (tmp_path / "s").iterdir()] == ["t.npz"]
    with np.load(tmp_path / "s" / "t.npz") as z:
        assert z["a"].tolist() == [0, 1, 2, 3, 4]


def test_deprecated_glue_warns():
    pytest.importorskip("crimm.Modeller", exc_type=ImportError)
    with pytest.warns(FutureWarning, match="deprecated"):
        import importlib
        import crimm_ml.crimm_utils as m
        importlib.reload(m)
