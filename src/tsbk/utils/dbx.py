import json
import os
import shutil
from pathlib import Path
from typing import Optional

import click
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.entities.model_registry import ModelVersion as MlflowModelVersion
from mlflow.tracking import MlflowClient

from tsbk import TSBK_DIR
from tsbk.utils import link_or_copy, parse_requirements_file


def compute_cache_path(models_uri: str) -> Path:
    """
    Outputs a unique path for caching the specified dbx model locally.

    Args:
        models_uri: the models uri to cache

    Returns:
        provided cache_dir with the model workspace, name, and version/tag appended
    """
    if not models_uri.startswith("models:/"):
        raise ValueError(f"{models_uri} is not a valid model URI")
    model_dir = models_uri.split("models:/")[-1]
    return TSBK_DIR.joinpath("cache/mlflow", model_dir)


def get_model_uri_parts(model_uri: str) -> tuple[str, str, str]:
    """Returns the catalog, schema, and model name from the models_uri"""
    if "@" in model_uri:
        scheme, model_path = model_uri.split("/")
        model_name, model_id = model_path.split("@")
        return scheme, model_name, model_id
    else:
        scheme, model_name, model_id = model_uri.split("/")
        return scheme, model_name, model_id


def set_registry_uri_from_models_uri(models_uri: str):
    """Sets the correct registry URI for the model. This is determined by checking if the model has a pattern like:
    models:/{catalog}.{schema_name}.{model_name}/{version}. This version corresponds to the unity catalog, otherwise it's
    the normal model registry."""
    _, model_name, _ = get_model_uri_parts(models_uri)
    try:
        _, _, _ = model_name.split(".")
        mlflow.set_registry_uri(f"databricks-uc")
    except ValueError:
        mlflow.set_registry_uri(None)  # noqa


def _download_model_by_run(model_uri: str, dst_path: str):
    """Downloads an MLFLow model from a URI"""
    # set_registry_uri_from_models_uri(model_uri)
    client = MlflowClient()

    models = find_matching_dbx_models(model_uri)

    if len(models) == 0:
        raise ValueError(f"Model {model_uri} not found")

    [model] = models
    path = model.source.split("/")[-1]

    model_path = client.download_artifacts(model.run_id, path=path)
    file_names = os.listdir(model_path)

    for file_name in file_names:
        shutil.move(os.path.join(model_path, file_name), dst_path)
    shutil.rmtree(model_path)


def _download_model_by_model_uri(model_uri: str, dst_path: str):
    """Downloads an MLFLow model from a URI"""
    # set_registry_uri_from_models_uri(model_uri)
    mlflow.artifacts.download_artifacts(model_uri, dst_path=dst_path)


def _download_model_from_dbx(model_uri: str, dst_path: str):
    try:
        _download_model_by_model_uri(model_uri, dst_path)
    except Exception as e1:
        click.secho(f"Failed to download model by model uri from databricks: {e1}", fg="red")
        click.secho("Trying to download model by run", fg="yellow")
        try:
            _download_model_by_run(model_uri, dst_path)
        except Exception as e2:
            click.secho(f"Failed to download model by run from databricks: {e2}", fg="red")
            shutil.rmtree(dst_path, ignore_errors=True)
            raise e1


def download_mlflow_model(
    origin_path: str,
    dst_path: Optional[str] = None,
) -> Path:
    """Downloads an MLFLow model to a specific directory in the ARTIFACT_CACHE_DIR where it can be used in Symlinks in Triton model directories. Symlinks are used to allow various Triton model directories to share the same model artifact.

    Args:
        origin_path: the model to download in mlflow model format - models:/<your-model>/<your-model-version>
        dst_path: Path where a symlink to the cached model artifact will be created for Triton to use when serving the model. If not provided, the model will be saved at a designated path in the ARTIFACT_CACHE_DIR and no symlink to this path will be created.

    Returns:
        The path to the model
    """
    # we can't use symlinking on DBX cluster, so if a dst_path is provided, we download the model directly to that path
    local_cache_path = compute_cache_path(origin_path)
    if not local_cache_path.exists() or not list(local_cache_path.glob("*")):
        local_cache_path.parent.mkdir(parents=True, exist_ok=True)
        click.secho(f"Downloading: {origin_path} -> {local_cache_path} (cache miss)", fg="blue")
        _download_model_from_dbx(origin_path, local_cache_path.as_posix())
    if dst_path is not None:
        click.secho(f"Creating hard link: {dst_path} -> {local_cache_path}", fg="blue")
        shutil.copytree(local_cache_path, dst_path, copy_function=link_or_copy, dirs_exist_ok=True)
        return Path(dst_path)
    else:
        return local_cache_path


def get_mlflow_model_requirements(model_uris: list[str]) -> Path:
    """
    Pulls artifacts for any MLFlow models provided in configuration and verifies that all requirements files are matching.

    Args:
        model_uris: List of model_uris to check requirements for.
    Returns:
        Path to requirements file, if MLFlow model requirements are matching.
    """
    models_paths = []
    for uri in model_uris:
        path_to_model = download_mlflow_model(uri, dst_path=None)
        models_paths.append(path_to_model)

    requirements_paths = [p.joinpath("requirements.txt") for p in models_paths]
    model_requirements = [parse_requirements_file(rp) for rp in requirements_paths]
    # raise exception if there is a mismatch across requirements for model versions
    if set.union(*model_requirements) != set.intersection(*model_requirements):
        raise ValueError(
            f"MLFlow model versions have mismatching requirements. models requirements: {model_requirements}"
        )
    return requirements_paths[-1]


def get_python_version_for_mlflow_models(model_uris: list[str]) -> str:
    """Checks the python version of all the mlflow models and returns the python version to use
    Args:
        model_uris: List of model_uris to check requirements for.
    Returns:
        the python version to use
    """
    models_paths = []
    for uri in model_uris:
        path_to_model = download_mlflow_model(uri, dst_path=None)
        models_paths.append(path_to_model)

    python_versions = set()
    for p in models_paths:
        try:
            if p.joinpath("conda.yaml").exists():
                with open(p.joinpath("conda.yaml"), "r") as fp:
                    python_versions.add(yaml.safe_load(fp)["dependencies"][0].split("=")[1])
            else:
                with open(p.joinpath("python_env.yaml"), "r") as fp:
                    python_versions.add(yaml.safe_load(fp)["python"])
        except FileNotFoundError:
            click.secho("python_env.yaml not found, checking MLmodel", fg="yellow")
            with open(p.joinpath("MLmodel"), "r") as fp:
                mlflow_config = yaml.safe_load(fp)
                python_versions.add(mlflow_config["flavors"]["python_function"]["python_version"])

    if len(python_versions) > 1:
        raise ValueError(
            f"MLFlow model versions have mismatching python versions. models python versions: {python_versions}"
        )

    return list(python_versions)[0]


def find_matching_dbx_models(model_uri: str) -> list[MlflowModelVersion]:
    """Identifies models versions in the MLFlow model registry that match the provided model_uri"""
    # set_registry_uri_from_models_uri(model_uri)
    client = MlflowClient()
    _, name, identifier = get_model_uri_parts(model_uri)
    if "@" not in model_uri and identifier.isdigit():  # Static model with version
        try:
            return [client.get_model_version(name, identifier)]
        except mlflow.exceptions.RestException as e:
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                return []
            raise e
    elif "@" not in model_uri and not identifier.isdigit():  # Model with stage name
        models = client.search_model_versions(filter_string=f"name = '{name}'")
        models = list(filter(lambda m: m.current_stage == identifier, models))
        return models
    elif "@" in model_uri:  # Model with alias
        try:
            return [client.get_model_version_by_alias(name, identifier)]
        except mlflow.exceptions.RestException as e:
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                return []
            raise e

    else:
        raise ValueError(f"Invalid model uri: {model_uri}")


def get_flavor_for_models(model_uris: list[str]) -> str:
    """Checks the models metadata and returns the triton backend to use
    Args:
        model_uris: List of model_uris to check requirements for.
    Returns:
        the triton backend to use for the models
    """
    models_paths = []
    for uri in model_uris:
        path_to_model = download_mlflow_model(uri, dst_path=None)
        models_paths.append(path_to_model)

    model_flavors = set()
    for p in models_paths:
        with open(p.joinpath("MLmodel"), "r") as fp:
            mlflow_config = yaml.safe_load(fp)
            flavors = tuple(mlflow_config["flavors"].keys())
            model_flavors.add(flavors)

    if len(model_flavors) > 1:
        raise ValueError(f"MLFlow model versions have mismatching model flavors. model flavors: {model_flavors}")

    model_flavors = model_flavors.pop()
    flavors_list = ["pytorch", "tensorflow", "onnx", "transformers", "sentence_transformers"]
    for flavor in flavors_list:
        if flavor in model_flavors:
            return flavor
    else:
        return "python"


def get_input_example_from_model(model_uri: str) -> dict[str, np.ndarray] | None:
    model_path = download_mlflow_model(model_uri)
    mlflow_config = yaml.safe_load(Path(model_path).joinpath("MLmodel").read_text())
    example_info = mlflow_config.get("saved_input_example_info")
    if example_info is None:
        return None

    input_example_path = Path(model_path).joinpath(example_info["artifact_path"])
    match example_info["type"]:
        case "dataframe":
            df = pd.read_json(input_example_path, orient=example_info["pandas_orient"])
            table_values = df.to_dict(orient="list")
            if example_info["pandas_orient"] == "split":  # example is actually a dataframe
                example_values = {k: np.array(v) for k, v in table_values.items()}
            else:  # example is just a list of values, see mlflow_toupper_model test fixture
                example_values = {k: np.array(v[0]) for k, v in table_values.items()}
            return example_values
        case "ndarray":
            result = json.loads(input_example_path.read_text())
            if example_info.get("format") == "tf-serving":
                if type(result["inputs"]) == dict:
                    return {k: np.array(v) for k, v in result["inputs"].items()}
                else:
                    return {"input0": np.array(result["inputs"])}
            else:
                return {k: np.array(v) for k, v in result.items()}

        case "json_object":
            result = json.loads(input_example_path.read_text())
            return {"input0": np.array(result)}
        case _:
            raise ValueError(f"Unsupported input example type: {example_info['type']}")


# TODO determine if this is needed, as it is not used in the current codebase
"""
def match_model_inputs(
    triton_client: tritonclient.http.InferenceServerClient | tritonclient.grpc.InferenceServerClient,
    model_name: str,
    inputs: dict[str, np.ndarray],
    headers: dict | None = None,
) -> dict[str, np.ndarray]:
    ret_dict = dict()

    if isinstance(triton_client, tritonclient.http.InferenceServerClient):
        model_inputs = triton_client.get_model_config(model_name, headers=headers)["input"]
    else:
        model_inputs = triton_client.get_model_config(model_name, as_json=True)["config"]["input"]

    # TODO provide more advanced matching capabilities
    if len(model_inputs) == 1:
        if len(inputs) > 1:
            raise ValueError(f"Model {model_name} only has one input, but {len(inputs)} were provided")
        input_name = model_inputs[0]["name"]
        input_type = model_inputs[0]["data_type"].split("_")[1]
        input_type = "BYTES" if input_type == "STRING" else input_type
        ret_dict[input_name] = inputs.pop(list(inputs)[0]).astype(triton_to_np_dtype(input_type))
    else:
        for input_config in model_inputs:
            input_name = input_config["name"]
            input_type = input_config["data_type"].split("_")[1]
            input_type = "BYTES" if input_type == "STRING" else input_type
            if input_name in inputs:
                ret_dict[input_name] = inputs.pop(input_name).astype(triton_to_np_dtype(input_type))
            else:
                raise ValueError(f"Input '{input_name}' not found in provided inputs for model {model_name}")

    return ret_dict
"""
