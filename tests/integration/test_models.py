import numpy as np
import pytest

import tsbk
from tsbk.testing import TritonModelRepoTestPlan


@pytest.fixture(scope="function")
def default_case():
    return tsbk.TestCase(
        inputs={
            "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "b": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        },
        expected_outputs={
            "add": np.array([5.0, 7.0, 9.0], dtype=np.float32),
            "sub": np.array([-3.0, -3.0, -3.0], dtype=np.float32),
        },
    )


@pytest.fixture(scope="function")
def default_case_batch():
    return tsbk.TestCase(
        inputs={
            "a": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "b": np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
        },
        expected_outputs={
            "add": np.array([[5.0, 7.0, 9.0]], dtype=np.float32),
            "sub": np.array([[-3.0, -3.0, -3.0]], dtype=np.float32),
        },
    )


def test_s3_pytorch_model(s3_pytorch_model, model_repo_dir, default_case):
    model_repo = tsbk.TritonModelRepo(
        name="test_s3_pytorch_model",
        path=model_repo_dir.joinpath("test_s3_pytorch_model"),
        models={
            "s3_pytorch_model": tsbk.TritonModel(
                backend="pytorch",
                inputs=[
                    tsbk.TritonDType(name="a", dtype=np.float32, dims=[-1]),
                    tsbk.TritonDType(name="b", dtype=np.float32, dims=[-1]),
                ],
                outputs=[
                    tsbk.TritonDType(name="add", dtype=np.float32, dims=[-1]),
                    tsbk.TritonDType(name="sub", dtype=np.float32, dims=[-1]),
                ],
                versions=[tsbk.TritonModelVersion(artifact_uri=s3_pytorch_model, test_cases=[default_case])],
                instance_group=[{"count": 3, "kind": "KIND_CPU"}],
            )
        },
    )
    TritonModelRepoTestPlan.from_dict(model_repo.create_test_plan().to_dict())

    model_repo.build()
    model_repo.run(detach=True)
    test_results = model_repo.test(model_repo.http_url) + model_repo.test(model_repo.grpc_url, grpc=True)
    model_repo.stop()

    assert len(test_results) == 2, "Expected two test case to run"
    for result in test_results:
        assert result.success, result.message


def test_s3_onnx_model(s3_onnx_model, model_repo_dir, default_case_batch):
    model_repo = tsbk.TritonModelRepo(
        name="test_s3_onnx_model",
        path=model_repo_dir.joinpath("test_s3_onnx_model"),
        models={
            "s3_onnx_model": tsbk.TritonModel(
                backend="onnxruntime",
                versions=[tsbk.TritonModelVersion(artifact_uri=s3_onnx_model)],
                test_cases=[default_case_batch],
            )
        },
    )
    TritonModelRepoTestPlan.from_dict(model_repo.create_test_plan().to_dict())

    model_repo.build()
    model_repo.run(detach=True)
    test_results = model_repo.test(model_repo.http_url) + model_repo.test(model_repo.grpc_url, grpc=True)
    model_repo.stop()

    assert len(test_results) == 2, "Expected two test case to run"
    for result in test_results:
        assert result.success, result.message


def test_python_model(assets_dir, model_repo_dir):
    test_case = tsbk.TestCase(
        inputs={
            "INPUT0": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "INPUT1": np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32),
        },
        expected_outputs={
            "OUTPUT0": np.array([5.0, 7.0, 9.0, 11.0], dtype=np.float32),
            "OUTPUT1": np.array([-3.0, -3.0, -3.0, -3.0], dtype=np.float32),
        },
    )
    model_repo = tsbk.TritonModelRepo(
        name="test_python_model",
        path=model_repo_dir.joinpath("test_python_model"),
        models={
            "python_model": tsbk.TritonModel(
                config_file=assets_dir.joinpath("python_model", "config.pbtxt"),
                requirements_file=assets_dir.joinpath("python_model", "requirements.txt"),
                python_version="3.11",
                versions=[
                    tsbk.TritonModelVersion(
                        python_model_file=assets_dir.joinpath("python_model", "model.py"),
                    )
                ],
                test_cases=[test_case],
            )
        },
    )
    TritonModelRepoTestPlan.from_dict(model_repo.create_test_plan().to_dict())

    model_repo.build()
    model_repo.run(detach=True)
    test_results = model_repo.test(model_repo.http_url) + model_repo.test(model_repo.grpc_url, grpc=True)
    assert len(test_results) == 2, "Expected two test case to run"
    for result in test_results:
        assert result.success, result.message
    model_repo.stop()


def test_mlflow_onnx_model(mlflow_onnx_model, model_repo_dir, default_case_batch):
    model_repo = tsbk.TritonModelRepo(
        name="test_mlflow_onnx_model",
        path=model_repo_dir.joinpath("test_mlflow_onnx_model"),
        models={
            "mlflow_onnx_model": tsbk.TritonModel(
                versions=[tsbk.TritonModelVersion(artifact_uri=mlflow_onnx_model)],
                test_cases=[default_case_batch],
            )
        },
    )
    TritonModelRepoTestPlan.from_dict(model_repo.create_test_plan().to_dict())

    model_repo.build()
    assert model_repo.models["mlflow_onnx_model"].versions[0].backend == "onnxruntime"

    model_repo.run(detach=True)
    test_results = model_repo.test(model_repo.http_url) + model_repo.test(model_repo.grpc_url, grpc=True)
    model_repo.stop()

    assert len(test_results) == 4, "Expected four test case to run"
    for result in test_results:
        assert result.success, result.message


def test_mlflow_python_model(mlflow_python_model, model_repo_dir):
    test_case = tsbk.TestCase(
        inputs={
            "a": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            "b": np.array([4.0, 5.0, 6.0], dtype=np.float64),
        },
        expected_outputs={
            "add": np.array([5.0, 7.0, 9.0], dtype=np.float64),
            "sub": np.array([-3.0, -3.0, -3.0], dtype=np.float64),
        },
    )
    model_repo = tsbk.TritonModelRepo(
        name="test_mlflow_python_model",
        path=model_repo_dir.joinpath("test_mlflow_python_model"),
        models={
            "mlflow_python_model": tsbk.TritonModel(
                versions=[tsbk.TritonModelVersion(artifact_uri=mlflow_python_model)],
                test_cases=[test_case],
            )
        },
    )
    TritonModelRepoTestPlan.from_dict(model_repo.create_test_plan().to_dict())

    model_repo.build()
    assert model_repo.models["mlflow_python_model"].versions[0].backend == "mlflow"

    model_repo.run(detach=True)
    test_results = model_repo.test(model_repo.http_url) + model_repo.test(model_repo.grpc_url, grpc=True)
    model_repo.stop()

    assert len(test_results) == 4, "Expected two test case to run"
    for result in test_results:
        assert result.success, result.message


def test_mlflow_pytorch_model(mlflow_pytorch_model, model_repo_dir, default_case):
    model_repo = tsbk.TritonModelRepo(
        name="test_mlflow_pytorch_model",
        path=model_repo_dir.joinpath("test_mlflow_pytorch_model"),
        models={
            "mlflow_pytorch_model": tsbk.TritonModel(
                versions=[tsbk.TritonModelVersion(artifact_uri=mlflow_pytorch_model)],
                test_cases=[default_case],
            )
        },
    )
    TritonModelRepoTestPlan.from_dict(model_repo.create_test_plan().to_dict())

    model_repo.build()
    assert model_repo.models["mlflow_pytorch_model"].versions[0].backend == "pytorch"

    model_repo.run(detach=True)
    test_results = model_repo.test(model_repo.http_url) + model_repo.test(model_repo.grpc_url, grpc=True)
    model_repo.stop()

    assert len(test_results) == 4, "Expected two test case to run"
    for result in test_results:
        assert result.success, result.message


def test_ensemble_model(assets_dir, model_repo_dir):
    test_case = tsbk.TestCase(
        inputs={
            "a": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        },
        expected_outputs={
            "c": np.array([1.01, 2.01, 3.01, 4.01], dtype=np.float32),
        },
        atol=1e-1,
        rtol=1e-1,
    )
    model_repo = tsbk.TritonModelRepo(
        name="test_python_model",
        path=model_repo_dir.joinpath("test_ensemble_model"),
        models={
            "add1": tsbk.TritonModel(
                config_file=assets_dir.joinpath("ensemble", "add1.pbtxt"),
                requirements_file=assets_dir.joinpath("ensemble", "requirements.txt"),
                python_version="3.11",
                versions=[
                    tsbk.TritonModelVersion(
                        python_model_file=assets_dir.joinpath("ensemble", "add1.py"),
                    )
                ],
            ),
            "sub1": tsbk.TritonModel(
                config_file=assets_dir.joinpath("ensemble", "sub1.pbtxt"),
                requirements_file=assets_dir.joinpath("ensemble", "requirements.txt"),
                python_version="3.11",
                versions=[
                    tsbk.TritonModelVersion(
                        python_model_file=assets_dir.joinpath("ensemble", "sub1.py"),
                    )
                ],
            ),
            "add_sub": tsbk.TritonModel(
                config_file=assets_dir.joinpath("ensemble", "ensemble.pbtxt"),
                test_cases=[test_case],
            ),
        },
    )
    TritonModelRepoTestPlan.from_dict(model_repo.create_test_plan().to_dict())

    model_repo.build()
    model_repo.run(detach=True)
    test_results = model_repo.test(model_repo.http_url) + model_repo.test(model_repo.grpc_url, grpc=True)
    model_repo.stop()

    assert len(test_results) == 6, "Expected two test case to run"
    for result in test_results:
        assert result.success, result.message


def test_decoupled_models(assets_dir, model_repo_dir):
    repeat_test_case = tsbk.TestCase(
        inputs={
            "IN": np.array([4, 2, 0, 1], dtype=np.int32),
            "DELAY": np.array([1, 2, 3, 4], dtype=np.uint32),
            "WAIT": np.array([5], dtype=np.uint32),
        },
        expected_outputs={
            "IDX": np.array([[0], [1], [2], [3]], dtype=np.uint32),
            "OUT": np.array([[4], [2], [0], [1]], dtype=np.int32),
        },
    )
    repeat_sum_test_case = tsbk.TestCase(
        inputs={
            "IN": np.array([4, 2, 0, 1], dtype=np.int32),
            "DELAY": np.array([1, 2, 3, 4], dtype=np.uint32),
            "WAIT": np.array([5], dtype=np.uint32),
        },
        expected_outputs={"SUM": np.array([7], dtype=np.int32)},
    )
    model_repo = tsbk.TritonModelRepo(
        name="test_decoupled_models",
        path=model_repo_dir.joinpath("test_decoupled_models"),
        models={
            "repeat": tsbk.TritonModel(
                config_file=assets_dir.joinpath("repeat", "config.pbtxt"),
                versions=[
                    tsbk.TritonModelVersion(
                        python_model_file=assets_dir.joinpath("repeat", "model.py"),
                    )
                ],
                test_cases=[repeat_test_case],
            ),
            "repeat_sum": tsbk.TritonModel(
                config_file=assets_dir.joinpath("repeat_sum", "config.pbtxt"),
                versions=[
                    tsbk.TritonModelVersion(
                        python_model_file=assets_dir.joinpath("repeat_sum", "model.py"),
                    )
                ],
                test_cases=[repeat_sum_test_case],
            ),
        },
    )
    TritonModelRepoTestPlan.from_dict(model_repo.create_test_plan().to_dict())

    model_repo.build()
    model_repo.run(detach=True)
    test_results = model_repo.test(model_repo.http_url) + model_repo.test(model_repo.grpc_url, grpc=True)
    model_repo.stop()

    assert len(test_results) == 4, "Expected four test case to run"
    for result in test_results:
        assert result.success or result.decoupled_with_http, result.message


def test_mlflow_model_with_batch_size_no_duplication(assets_dir, mlflow_onnx_model, model_repo_dir):
    """Test that max_batch_size appears only once when using an MLflow model with a config that has max_batch_size."""
    model_repo = tsbk.TritonModelRepo(
        name="test_mlflow_with_batch",
        path=model_repo_dir.joinpath("test_mlflow_with_batch"),
        models={
            "mlflow_model_with_batch": tsbk.TritonModel(
                config_file=assets_dir.joinpath("mlflow", "with_batch.pbtxt"),
                versions=[tsbk.TritonModelVersion(artifact_uri=mlflow_onnx_model)],
            )
        },
    )

    model_repo.build()

    # Read the generated config.pbtxt
    config_path = model_repo_dir.joinpath("test_mlflow_with_batch", "mlflow_model_with_batch", "config.pbtxt")
    config_text = config_path.read_text()

    # Count occurrences of max_batch_size
    max_batch_size_count = config_text.count("max_batch_size")

    assert (
        max_batch_size_count == 1
    ), f"Expected max_batch_size to appear once, but found {max_batch_size_count} occurrences in:\n{config_text}"


def test_mlflow_model_without_batch_size_no_duplication(assets_dir, mlflow_onnx_model, model_repo_dir):
    """Test that max_batch_size appears only once when using an MLflow model with a config that doesn't have max_batch_size."""
    model_repo = tsbk.TritonModelRepo(
        name="test_mlflow_without_batch",
        path=model_repo_dir.joinpath("test_mlflow_without_batch"),
        models={
            "mlflow_model_without_batch": tsbk.TritonModel(
                config_file=assets_dir.joinpath("mlflow", "without_batch.pbtxt"),
                versions=[tsbk.TritonModelVersion(artifact_uri=mlflow_onnx_model)],
            )
        },
    )

    model_repo.build()

    # Read the generated config.pbtxt
    config_path = model_repo_dir.joinpath("test_mlflow_without_batch", "mlflow_model_without_batch", "config.pbtxt")
    config_text = config_path.read_text()

    # Count occurrences of max_batch_size
    max_batch_size_count = config_text.count("max_batch_size")

    assert (
        max_batch_size_count == 1
    ), f"Expected max_batch_size to appear once, but found {max_batch_size_count} occurrences in:\n{config_text}"
