from uuid import uuid4

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from tsbk import TritonModel, TritonModelRepo, TritonModelVersion
from tsbk.spec import TritonModelRepoSpec, TritonModelSpec, TritonModelVersionSpec
from tsbk.utils.dbx import compute_cache_path as compute_mlflow_cache_path
from tsbk.utils.s3 import compute_cache_path as compute_s3_cache_path


def test_repo_cache_bust_rotates_s3_artifact_cache(s3_bucket, s3_client, tmp_path, monkeypatch):
    monkeypatch.setattr("tsbk.utils.s3.TSBK_DIR", tmp_path)
    s3_client.put_object(Bucket=s3_bucket, Key="cache-bust/model.onnx", Body=b"model")
    artifact_uri = f"s3://{s3_bucket}/cache-bust/model.onnx"

    spec = TritonModelRepoSpec(
        name="cache-bust-s3",
        cache_bust="release/one",
        models={
            "model": TritonModelSpec(
                backend="onnxruntime",
                versions=[TritonModelVersionSpec(artifact_uri=artifact_uri)],
            )
        },
    )
    first_repo = TritonModelRepo(path=tmp_path / "repo-one", **spec.model_dump())
    first_repo.build()

    spec.cache_bust = "release/two"
    second_repo = TritonModelRepo(path=tmp_path / "repo-two", **spec.model_dump())
    second_repo.build()

    first_cache_path = compute_s3_cache_path(artifact_uri, cache_bust="release/one")
    second_cache_path = compute_s3_cache_path(artifact_uri, cache_bust="release/two")
    assert first_cache_path != second_cache_path
    assert first_cache_path.read_bytes() == second_cache_path.read_bytes()
    assert first_repo.cache_bust == "release/one"
    assert second_repo.cache_bust == "release/two"


def test_repo_cache_bust_rotates_mlflow_artifact_cache(mlflow_tracking_uri, tmp_path, monkeypatch):
    monkeypatch.setattr("tsbk.utils.dbx.TSBK_DIR", tmp_path)
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    class AddOneModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input + 1

    registered_name = f"cache_bust_{uuid4().hex}"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            "model",
            python_model=AddOneModel(),
            registered_model_name=registered_name,
            input_example=pd.DataFrame({"value": [1.0]}),
        )
    [registered_version] = MlflowClient().search_model_versions(f"name = '{registered_name}'")
    mlflow_model = f"models:/{registered_name}/{registered_version.version}"

    def create_repo(cache_bust: str, path: str):
        return TritonModelRepo(
            name="cache-bust-mlflow",
            path=tmp_path / path,
            cache_bust=cache_bust,
            models={
                "model": TritonModel(
                    versions=[TritonModelVersion(artifact_uri=mlflow_model)],
                )
            },
        )

    first_repo = create_repo("release/one", "repo-one")
    second_repo = create_repo("release/two", "repo-two")

    first_cache_path = compute_mlflow_cache_path(mlflow_model, cache_bust="release/one")
    second_cache_path = compute_mlflow_cache_path(mlflow_model, cache_bust="release/two")
    assert first_cache_path != second_cache_path
    assert first_cache_path.joinpath("MLmodel").exists()
    assert second_cache_path.joinpath("MLmodel").exists()
    assert first_repo.models["model"].mlflow_model_paths == [first_cache_path]
    assert second_repo.models["model"].mlflow_model_paths == [second_cache_path]
