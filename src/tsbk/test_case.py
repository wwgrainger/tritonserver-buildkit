import traceback
from dataclasses import dataclass

import numpy as np

from tsbk.triton_grpc_model_client import TritonGrpcModelClient
from tsbk.triton_http_model_client import TritonHttpModelClient
from tsbk.types import NumpyArray


@dataclass
class TestResult:
    """A class to hold the result of a test case execution."""

    success: bool
    model_name: str
    model_version: int
    message: str | None = None
    exception: BaseException | None = None
    traceback: str | None = None
    missing_test: bool = False
    decoupled_with_http: bool = False


class TestCase:
    def __init__(
        self,
        inputs: dict[str, np.ndarray | NumpyArray | dict],
        expected_outputs: dict[str, np.ndarray | NumpyArray | dict],
        allow_nan: bool = False,
        allow_inf: bool = False,
        rtol: float = 1.0e-5,
        atol: float = 1.0e-8,
    ):
        self.inputs = inputs
        for k, v in self.inputs.items():
            if isinstance(v, NumpyArray):
                self.inputs[k] = v.to_numpy()
            elif isinstance(v, dict):
                self.inputs[k] = NumpyArray(**v).to_numpy()

        self.expected_outputs = expected_outputs
        for k, v in self.expected_outputs.items():
            if isinstance(v, NumpyArray):
                self.expected_outputs[k] = v.to_numpy()
            elif isinstance(v, dict):
                self.expected_outputs[k] = NumpyArray(**v).to_numpy()

        self.allow_nan = allow_nan
        self.allow_inf = allow_inf
        self.rtol = rtol
        self.atol = atol

    def run(self, model_client: TritonHttpModelClient | TritonGrpcModelClient, decoupled: bool = False) -> TestResult:
        try:
            if decoupled:
                assert isinstance(
                    model_client, TritonGrpcModelClient
                ), "Decoupled inference only supported with GRPC client"
                outputs = model_client.infer_decoupled(self.inputs)
            else:
                outputs = model_client.infer(self.inputs)
            if set(self.expected_outputs.keys()) > set(outputs.keys()):
                return TestResult(
                    success=False,
                    model_name=model_client.model_name,
                    model_version=model_client.model_version,
                    message=f"Expected outputs {self.expected_outputs.keys()} not found in model outputs {outputs.keys()}",
                )
            for tensor_name, output in outputs.items():
                expected = self.expected_outputs.get(tensor_name)
                self.validate_output(output, expected, tensor_name, self.allow_nan, self.allow_inf, self.rtol, self.atol)
        except Exception as e:
            return TestResult(
                success=False,
                model_name=model_client.model_name,
                model_version=model_client.model_version,
                message=f"Inference failed for {model_client.model_name}:{model_client.model_version}",
                exception=e,
                traceback=traceback.format_exc(),
            )
        return TestResult(
            success=True,
            model_name=model_client.model_name,
            model_version=model_client.model_version,
            message=f"Inference succeeded for {model_client.model_name}:{model_client.model_version}",
        )

    @staticmethod
    def validate_output(
        output: np.ndarray,
        expected: np.ndarray | None,
        tensor_name: str,
        allow_nan: bool = False,
        allow_inf: bool = False,
        rtol: float = 1.0e-5,
        atol: float = 1.0e-8,
    ):
        """Standard function for validating model outputs. Raises a ValueError if the output is not as expected.

        Args:
            output: The model output to validate
            expected: The expected output. If None, only NaNs and Infs will be checked in the output
            tensor_name: The name of the tensor being validated
            allow_nan: Whether to allow NaNs in the output
            allow_inf: Whether to allow Infs in the output
            rtol: The relative tolerance to use for comparing floating point values
            atol: The absolute tolerance to use for comparing floating point values

        Raises:
            ValueError: If the output is not as expected
        """

        if output.dtype != object and np.isnan(output).any() and not allow_nan:
            raise ValueError(f"Output {tensor_name} contains NaNs")
        if output.dtype != object and np.isinf(output).any() and not allow_inf:
            raise ValueError(f"Output {tensor_name} contains Infs")

        if expected is not None:
            if output.shape != expected.shape:
                raise ValueError(
                    f"Output {tensor_name} has incorrect shape. Expected {expected.shape}, got {output.shape}"
                )

            if output.dtype != expected.dtype:
                raise ValueError(
                    f"Output {tensor_name} has incorrect dtype. Expected {expected.dtype}, got {output.dtype}"
                )

            if output.dtype == object:
                if not np.all(output == expected):
                    raise ValueError(f"Output {tensor_name} has incorrect values. Expected {expected}, got {output}")
            else:
                if not np.allclose(output, expected, rtol=rtol, atol=atol, equal_nan=allow_nan):
                    raise ValueError(f"Output {tensor_name} has incorrect values. Difference: {output - expected}")

    def to_dict(self) -> dict:
        """Convert the test case to a dictionary representation."""
        return {
            "inputs": {k: NumpyArray.from_numpy(v).to_dict() for k, v in self.inputs.items()},
            "expected_outputs": {k: NumpyArray.from_numpy(v).to_dict() for k, v in self.expected_outputs.items()},
            "allow_nan": self.allow_nan,
            "allow_inf": self.allow_inf,
            "rtol": self.rtol,
            "atol": self.atol,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TestCase":
        """Create a TestCase instance from a dictionary representation."""
        return cls(
            inputs=d["inputs"],
            expected_outputs=d["expected_outputs"],
            allow_nan=d.get("allow_nan", False),
            allow_inf=d.get("allow_inf", False),
            rtol=d.get("rtol", 1.0e-5),
            atol=d.get("atol", 1.0e-8),
        )
