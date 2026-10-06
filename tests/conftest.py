import os
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def onnx_model_path():
    """The ONNX model to test against: `$CRIMM_ML_TEST_MODEL`, or skip."""
    path = os.environ.get("CRIMM_ML_TEST_MODEL")
    if not path:
        pytest.skip("set CRIMM_ML_TEST_MODEL to an rf_loop_dynamic_tp_ort.onnx to run this test")
    path = Path(path)
    if not path.exists():
        pytest.fail(f"CRIMM_ML_TEST_MODEL={path} does not exist")
    return path


@pytest.fixture(scope="session")
def pdb_lines():
    return (DATA / "2KL8.pdb").read_text().splitlines(keepends=True)
