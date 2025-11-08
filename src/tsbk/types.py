import numpy as np
from tritonclient.utils import np_to_triton_dtype


class TritonDType:
    TRITON_DTYPES = {
        "BOOL",
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
        "FP16",
        "FP32",
        "FP64",
        "STRING",
    }

    def __init__(self, name: str, dtype: np.dtype | str, dims: list[int]):
        self.name = name
        if dtype not in self.TRITON_DTYPES:
            if isinstance(dtype, str):
                dtype = np.dtype(dtype)

            self.dtype = np_to_triton_dtype(dtype)
            self.dtype = "STRING" if self.dtype == "BYTES" else self.dtype
        else:
            self.dtype = dtype

        assert (
            self.dtype in self.TRITON_DTYPES
        ), f"Invalid Triton dtype: {self.dtype}. Must be one of {self.TRITON_DTYPES}"
        self.dims = dims


class NumpyArray:
    def __init__(self, dtype: np.dtype | str, dims: list[int], data: list):
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)
        self.dtype = dtype
        self.dims = dims
        self.data = data

    def to_numpy(self) -> np.ndarray:
        return np.array(self.data, dtype=self.dtype).reshape(self.dims)

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> "NumpyArray":
        return cls(dtype=array.dtype, dims=list(array.shape), data=array.flatten().tolist())

    def to_dict(self) -> dict:
        return {
            "dtype": str(self.dtype),
            "dims": self.dims,
            "data": self.data,
        }
