import numpy as np

from tsbk.types import TritonDType


def test_triton_dtype():
    dtype = TritonDType("name", "float32", [1, 2, 3])
    assert dtype.dtype == "FP32"

    dtype = TritonDType("name", np.float32, [1, 2, 3])
    assert dtype.dtype == "FP32"

    dtype = TritonDType("name", "FP32", [1, 2, 3])
    assert dtype.dtype == "FP32"

    dtype = TritonDType("name", np.object_, [1, 2, 3])
    assert dtype.dtype == "STRING"

    dtype = TritonDType("name", np.dtype("object"), [1, 2, 3])
    assert dtype.dtype == "STRING"
