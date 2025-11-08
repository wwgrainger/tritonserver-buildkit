import pytest

from tsbk.utils.triton import parse_mlflow_signature, parse_mlflow_signature_value


@pytest.mark.parametrize(
    "value, expected",
    [
        ({"type": "string"}, (None, "STRING", [-1], 0)),
        (
            {"type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 3, 224, 224]}},
            (None, "FP32", [3, 224, 224], -1),
        ),
        ({"type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 1000]}}, (None, "FP32", [1000], -1)),
        ({"type": "tensor", "tensor-spec": {"dtype": "object", "shape": [-1]}}, (None, "STRING", [-1], 0)),
        ({"type": "tensor", "tensor-spec": {"dtype": "int64", "shape": [-1, 6]}}, (None, "INT64", [6], -1)),
    ],
)
def test_parse_mlflow_signature_value(value, expected):
    assert parse_mlflow_signature_value(value) == expected


def test_parse_simple_mlflow_signature():
    input_sig = {"type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 3, 224, 224]}}
    output_sig = {"type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 1000]}}
    mlflow_signature = ([input_sig], [output_sig])
    batch_size, inputs, outputs = parse_mlflow_signature(mlflow_signature)
    assert batch_size == -1
    assert inputs == {"input0": ("FP32", [3, 224, 224], -1)}
    assert outputs == {"output0": ("FP32", [1000], -1)}


def test_parse_mlflow_signature_with_batch_size_change():
    input_sig = {"type": "tensor", "tensor-spec": {"dtype": "object", "shape": [-1]}}
    output_sig = {"type": "tensor", "tensor-spec": {"dtype": "int64", "shape": [-1, 6]}}
    mlflow_signature = ([input_sig], [output_sig])
    batch_size, inputs, outputs = parse_mlflow_signature(mlflow_signature)
    assert batch_size == 0
    assert inputs == {"input0": ("STRING", [-1], 0)}
    assert outputs == {"output0": ("INT64", [-1, 6], 0)}


def test_parse_mlflow_signature_with_compatible_batch_sizes():
    input_sig = {"type": "tensor", "tensor-spec": {"dtype": "object", "shape": [1, 100]}}
    output_sig = {"type": "tensor", "tensor-spec": {"dtype": "int64", "shape": [-1, 6]}}
    mlflow_signature = ([input_sig], [output_sig])
    batch_size, inputs, outputs = parse_mlflow_signature(mlflow_signature)
    assert batch_size == 1
    assert inputs == {"input0": ("STRING", [100], 1)}
    assert outputs == {"output0": ("INT64", [6], 1)}


def test_parse_mlflow_signature_with_compatible_normal_batch_sizes():
    input_sig = {"type": "tensor", "tensor-spec": {"dtype": "object", "shape": [1, 100]}}
    output_sig = {"type": "tensor", "tensor-spec": {"dtype": "int64", "shape": [2, 6]}}
    mlflow_signature = ([input_sig], [output_sig])
    batch_size, inputs, outputs = parse_mlflow_signature(mlflow_signature)
    assert batch_size == 1
    assert inputs == {"input0": ("STRING", [100], 1)}
    assert outputs == {"output0": ("INT64", [6], 1)}
