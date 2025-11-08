from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from tsbk.utils import link_or_copy
from tsbk.utils.dbx import (
    compute_cache_path,
    download_mlflow_model,
    get_flavor_for_models,
    get_mlflow_model_requirements,
    get_python_version_for_mlflow_models,
    set_registry_uri_from_models_uri,
)


def test_compute_cache_path():
    cache_dir = Path("/my/cache/path")
    with patch("tsbk.utils.dbx.TSBK_DIR", cache_dir):
        assert compute_cache_path(
            "models:/my-model/1",
        ) == Path("/my/cache/path/cache/mlflow/my-model/1")


def test_compute_cache_path_bad():
    cache_dir = Path("/my/cache/path")
    with patch("tsbk.utils.dbx.TSBK_DIR", cache_dir):
        with pytest.raises(ValueError) as excinfo:
            compute_cache_path("s3://some-bucket/some-key")

    assert "s3://some-bucket/some-key is not a valid model URI" in str(excinfo.value)


@patch("tsbk.utils.dbx.compute_cache_path")
@patch("tsbk.utils.dbx._download_model_from_dbx")
def test_download_mlflow_model_no_dst_path(mock_download, mock_compute_cache_path):
    r_mock = MagicMock()
    r_mock.exists.return_value = False
    mock_compute_cache_path.return_value = r_mock
    p = download_mlflow_model("models:/model/1")
    assert p == r_mock
    assert r_mock.parent.mkdir.call_args == call(parents=True, exist_ok=True)
    assert mock_download.call_args == call("models:/model/1", r_mock.as_posix.return_value)


@patch("tsbk.utils.dbx.compute_cache_path")
@patch("tsbk.utils.dbx._download_model_from_dbx")
@patch("tsbk.utils.dbx.shutil.copytree")
def test_download_mlflow_model_w_dst_path(mock_copy, mock_download, mock_compute_cache_path, tmp_path):
    r_mock = MagicMock()
    r_mock.exists.return_value = True
    r_mock.glob.return_value = ["some-file"]
    mock_compute_cache_path.return_value = r_mock
    p = download_mlflow_model("models:/model/1", dst_path="my/dest/path")
    assert p == Path("my/dest/path")
    assert r_mock.parent.mkdir.call_count == 0
    assert mock_download.call_count == 0
    assert mock_copy.call_args == call(r_mock, "my/dest/path", copy_function=link_or_copy, dirs_exist_ok=True)


@pytest.fixture()
def matching_mlflow_models(tmp_path):
    model1, model2 = tmp_path.joinpath("model1"), tmp_path.joinpath("model2")
    model1.mkdir()
    model2.mkdir()
    model1.joinpath("requirements.txt").write_text("req1==1\nreq2==2")
    model2.joinpath("requirements.txt").write_text("req1==1\nreq2==2")
    model1.joinpath("python_env.yaml").write_text('python: "3.8"')
    model2.joinpath("python_env.yaml").write_text('python: "3.8"')
    model1.joinpath("MLmodel").write_text(
        'signature: {"inputs": "[\\"test input\\"]", "outputs": "[\\"test output\\"]"}\nflavors: {"python": {}}'
    )
    model2.joinpath("MLmodel").write_text(
        'signature: {"inputs": "[\\"test input\\"]", "outputs": "[\\"test output\\"]"}\nflavors: {"python": {}}'
    )
    return model1, model2


@pytest.fixture()
def mismatching_mlflow_models(tmp_path):
    model1, model2 = tmp_path.joinpath("model1"), tmp_path.joinpath("model2")
    model1.mkdir()
    model2.mkdir()
    model1.joinpath("requirements.txt").write_text("req1==1\nreq2==2")
    model2.joinpath("requirements.txt").write_text("req1==2\nreq2==3")
    model1.joinpath("python_env.yaml").write_text('python: "3.9.5"')
    model2.joinpath("python_env.yaml").write_text('python: "3.10.5"')
    model1.joinpath("MLmodel").write_text(
        'signature: {"inputs": "[\\"test input1\\"]", "outputs": "[\\"test output1\\"]"}\nflavors: {"python": {}}'
    )
    model2.joinpath("MLmodel").write_text(
        'signature: {"inputs": "[\\"test input2\\"]", "outputs": "[\\"test output2\\"]"}\nflavors: {"pytorch": {}}'
    )
    return model1, model2


@patch("tsbk.utils.dbx.download_mlflow_model")
def test_get_mlflow_model_requirements(download_mlflow_patch, matching_mlflow_models):
    download_mlflow_patch.side_effect = matching_mlflow_models
    path = get_mlflow_model_requirements([MagicMock(), MagicMock()])
    assert path == matching_mlflow_models[1].joinpath("requirements.txt")


@patch("tsbk.utils.dbx.download_mlflow_model")
def test_get_requirements_for_mlflow_models_mismatching(download_mlflow_patch, mismatching_mlflow_models):
    download_mlflow_patch.side_effect = mismatching_mlflow_models
    with pytest.raises(ValueError) as e:
        get_mlflow_model_requirements([MagicMock(), MagicMock()])


@patch("tsbk.utils.dbx.download_mlflow_model")
def test_get_python_version_for_mlflow_models(download_mlflow_patch, matching_mlflow_models):
    download_mlflow_patch.side_effect = matching_mlflow_models
    python_version = get_python_version_for_mlflow_models([MagicMock(), MagicMock()])
    assert python_version == "3.8"


@patch("tsbk.utils.dbx.download_mlflow_model")
def test_get_python_version_for_mlflow_models_mismatching(download_mlflow_patch, mismatching_mlflow_models):
    download_mlflow_patch.side_effect = mismatching_mlflow_models
    with pytest.raises(ValueError) as e:
        get_python_version_for_mlflow_models([MagicMock(), MagicMock()])


@patch("tsbk.utils.dbx.download_mlflow_model")
def test_get_model_flavor_for_mlflow_models_matching(download_mlflow_patch, matching_mlflow_models):
    download_mlflow_patch.side_effect = matching_mlflow_models
    backend = get_flavor_for_models([MagicMock(), MagicMock()])
    assert backend == "python"


@patch("tsbk.utils.dbx.download_mlflow_model")
def test_get_model_flavor_for_mlflow_models_mismatching(download_mlflow_patch, mismatching_mlflow_models):
    download_mlflow_patch.side_effect = mismatching_mlflow_models
    with pytest.raises(ValueError) as e:
        get_flavor_for_models([MagicMock(), MagicMock()])


@patch("tsbk.utils.dbx.mlflow")
def test_set_registry_uri_no_uc(mlflow_patch):
    set_registry_uri_from_models_uri("models:/model/1")
    assert mlflow_patch.set_registry_uri.call_args == call(None)


@patch("tsbk.utils.dbx.mlflow")
def test_set_registry_uri_uc(mlflow_patch):
    set_registry_uri_from_models_uri("models:/main.default.model/1")
    assert mlflow_patch.set_registry_uri.call_args == call("databricks-uc")
