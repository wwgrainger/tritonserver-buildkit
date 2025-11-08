import subprocess
from pathlib import Path

import boto3
import mlflow
import numpy as np
import onnx
import pandas as pd
import pytest
import torch
import torch.onnx
from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import Schema, TensorSpec


@pytest.fixture(scope="session", autouse=True)
def cleanup_docker_images():
    # Get currently running docker containers
    result = subprocess.run(["docker", "ps", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    container_ids = result.stdout.strip().split("\n")
    yield
    # Cleanup new containers
    result = subprocess.run(["docker", "ps", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    new_container_ids = result.stdout.strip().split("\n")
    for container_id in set(new_container_ids) - set(container_ids):
        print("Stopping container:", container_id)
        subprocess.run(["docker", "stop", container_id], check=False)
        subprocess.run(["docker", "rm", container_id], check=False)


@pytest.fixture(scope="session")
def s3_client():
    return boto3.client(
        "s3",
        region_name="us-east-2",
        endpoint_url="http://localhost:4566",
    )


@pytest.fixture(scope="session")
def s3_bucket(s3_client):
    bucket_name = "test-bucket"
    s3_r = boto3.resource("s3", endpoint_url="http://localhost:4566")
    try:
        bucket = s3_r.Bucket(bucket_name)
        bucket.objects.all().delete()
        bucket.delete()
    except Exception:
        pass
    s3_client.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": "us-east-2"},
    )
    return bucket_name


@pytest.fixture(scope="session")
def model_repo_dir():
    return Path(__file__).parents[2] / "model_repos"


@pytest.fixture(scope="session")
def assets_dir():
    return Path(__file__).parents[1] / "assets"


@pytest.fixture(scope="function")
def mlflow_tracking_uri():
    return f"file:{Path(__file__).parents[2]}/mlruns"


@pytest.fixture(scope="function")
def s3_pytorch_model(s3_bucket, s3_client, tmp_path):
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return a + b, a - b

    model = Model()
    model_path = tmp_path / "model.pt"
    torch.jit.save(torch.jit.script(model), model_path)
    s3_client.upload_file(
        Filename=str(model_path),
        Bucket=s3_bucket,
        Key="pytorch_model/model.pt",
    )
    return f"s3://{s3_bucket}/pytorch_model/model.pt"


@pytest.fixture(scope="function")
def mlflow_pytorch_model(mlflow_tracking_uri, s3_pytorch_model, tmp_path, s3_client):
    s3_client.download_file(
        Bucket=s3_pytorch_model.split("/")[2],
        Key="/".join(s3_pytorch_model.split("/")[3:]),
        Filename=str(tmp_path / "model.pt"),
    )
    model = torch.load(str(tmp_path / "model.pt"), weights_only=False)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    input_schema = Schema(
        [
            TensorSpec(np.dtype(np.float32), (-1,), name="a"),
            TensorSpec(np.dtype(np.float32), (-1,), name="b"),
        ]
    )
    output_schema = Schema(
        [
            TensorSpec(np.dtype(np.float32), (-1,), name="add"),
            TensorSpec(np.dtype(np.float32), (-1,), name="sub"),
        ]
    )
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    with mlflow.start_run() as run:
        model = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="pytorch_model",
            registered_model_name="mlflow_pytorch_model",
            input_example={
                "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "b": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            },
            signature=signature,
        )
    return f"models:/mlflow_pytorch_model/{model.registered_model_version}"


@pytest.fixture(scope="function")
def s3_onnx_model(s3_bucket, s3_client, tmp_path):
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return {
                "add": a + b,
                "sub": a - b,
            }

    model = Model()
    model_path = tmp_path / "model.onnx"
    dummy_input = (torch.randn(1, 3), torch.randn(1, 3))
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        input_names=["a", "b"],
        output_names=["add", "sub"],
        dynamic_axes={
            "a": {0: "batch_size"},
            "b": {0: "batch_size"},
            "add": {0: "batch_size"},
            "sub": {0: "batch_size"},
        },
    )

    s3_client.upload_file(
        Filename=str(model_path),
        Bucket=s3_bucket,
        Key="onnx_model/model.onnx",
    )
    return f"s3://{s3_bucket}/onnx_model/model.onnx"


@pytest.fixture(scope="function")
def mlflow_onnx_model(mlflow_tracking_uri, s3_onnx_model, tmp_path, s3_client):
    s3_client.download_file(
        Bucket=s3_onnx_model.split("/")[2],
        Key="/".join(s3_onnx_model.split("/")[3:]),
        Filename=str(tmp_path / "model.onnx"),
    )
    onnx_model = onnx.load_model(str(tmp_path / "model.onnx"))
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    with mlflow.start_run() as run:
        model = mlflow.onnx.log_model(
            onnx_model,
            artifact_path="onnx_model",
            registered_model_name="mlflow_onnx_model",
            input_example={
                "a": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                "b": np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
            },
        )
    return f"models:/mlflow_onnx_model/{model.registered_model_version}"


@pytest.fixture(scope="session")
def mlflow_python_model():
    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input: pd.DataFrame, params=None):
            """Combine two columns"""
            return pd.DataFrame({"add": model_input["a"] + model_input["b"], "sub": model_input["a"] - model_input["b"]})

    model = Model()
    input_example = pd.DataFrame({"a": [1.0, 3.0], "b": [2.0, 4.0]})
    signature = infer_signature(input_example, model.predict(None, input_example))
    with mlflow.start_run() as run:
        model = mlflow.pyfunc.log_model(
            "pyfunc-model",
            python_model=model,
            input_example=input_example,
            signature=signature,
            registered_model_name="mlflow_python_model",
        )
    return f"models:/mlflow_python_model/{model.registered_model_version}"
