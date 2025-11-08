import numpy as np
from tritonclient.utils import np_to_triton_dtype


def parse_mlflow_signature_value(value: dict) -> tuple[str | None, str, list[int], int]:
    """Parses a MLFlow signature value into a tuple of (name, type, shape, batch size)

    Args:
        value: MLFlow signature value

    Returns:
        Tuple of (name, type, shape, batch size)
    """
    if value["type"] == "tensor":
        dtype = np_to_triton_dtype(np.dtype(value["tensor-spec"]["dtype"]))
        dtype = "STRING" if dtype == "BYTES" else dtype
        shape = value["tensor-spec"]["shape"]
        if len(shape) > 1:
            dims = shape[1:]
            batch_size = shape[0]
        else:
            dims = shape
            batch_size = 0
    elif value["type"] == "array":
        dtype = "STRING"
        dims = [-1]
        batch_size = 0
    elif value["type"] == "integer":
        dtype = "INT32"
        dims = [-1]
        batch_size = 0
    else:
        dtype = "STRING" if value["type"] == "string" else np_to_triton_dtype(np.dtype(value["type"]))
        dims = [-1]
        batch_size = 0

    return value.get("name"), f"{dtype}", dims, batch_size


def parse_mlflow_signature(mlflow_signature: tuple[list, list]) -> tuple[int, dict, dict]:
    """Parses an MLFlow model signature into Triton input and output schemas

    Args:
        mlflow_signature: MLFlow model signature

    Returns:
        Tuple of max batch size, input schema, and output schema
    """
    batch_sizes = set()
    inputs = dict()
    for i, model_input in enumerate(mlflow_signature[0]):
        name, dtype, dims, batch_size = parse_mlflow_signature_value(model_input)
        if name is None:
            name = f"input{i}"
        inputs[name] = dtype, dims, batch_size
        batch_sizes.add(batch_size)

    outputs = dict()
    for i, model_output in enumerate(mlflow_signature[1]):
        name, dtype, dims, batch_size = parse_mlflow_signature_value(model_output)
        if name is None:
            name = f"output{i}"
        outputs[name] = dtype, dims, batch_size
        batch_sizes.add(batch_size)

    if batch_sizes == {-1, 0}:
        for name, (dtype, dims, batch_size) in inputs.items():
            if batch_size == -1:
                inputs[name] = dtype, [-1] + dims, 0
        for name, (dtype, dims, batch_size) in outputs.items():
            if batch_size == -1:
                outputs[name] = dtype, [-1] + dims, 0
        batch_sizes.remove(-1)

    if len(batch_sizes) > 1:
        if 0 in batch_sizes:
            raise ValueError(f"Inconsistent batch sizes: {batch_sizes}")
        if -1 in batch_sizes:
            batch_sizes.remove(-1)
        batch_size = min(batch_sizes)
        for name, (dtype, dims, _) in inputs.items():
            inputs[name] = dtype, dims, batch_size
        for name, (dtype, dims, _) in outputs.items():
            outputs[name] = dtype, dims, batch_size
        batch_sizes = {batch_size}

    return batch_sizes.pop(), inputs, outputs


def python_version_for_triton_version(triton_version: str) -> str:
    """Returns the default Python version for a given Triton version.

    Args:
        triton_version: The Triton version.

    Returns:
        The default Python version for the given Triton version.
    """
    version_no = float(triton_version)
    if version_no >= 24.11:
        return "3.12"
    else:
        return "3.10"
