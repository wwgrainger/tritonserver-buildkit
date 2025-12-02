import shutil
from pathlib import Path
from typing import Any

import click
import yaml
from mlflow_backend_utils.sdk import build_config
from mlflow_backend_utils.utils import parse_mlflow_signature

from tsbk.model_version import TritonModelVersion
from tsbk.test_case import TestCase
from tsbk.testing import TritonModelTestPlan
from tsbk.types import TritonDType
from tsbk.utils import link_or_copy, parse_python_version
from tsbk.utils.conda import build_conda_env
from tsbk.utils.dbx import (
    download_mlflow_model,
    find_matching_dbx_models,
    get_flavor_for_models,
    get_mlflow_model_requirements,
    get_python_version_for_mlflow_models,
)
from tsbk.utils.pbtxt import parse_pbtxt
from tsbk.utils.stubs import build_triton_stub
from tsbk.utils.triton import python_version_for_triton_version

BACKENDS_EXCLUDE_SET_SCHEMA = ["onnxruntime"]


class TritonModel:
    def __init__(
        self,
        *,
        versions: list[TritonModelVersion | dict] | None = None,
        backend: str | None = None,
        platform: str | None = None,
        max_batch_size: int | None = None,
        inputs: list[TritonDType | dict] | None = None,
        outputs: list[TritonDType | dict] | None = None,
        config: str | None = None,
        config_file: str | None = None,
        python_version: str | None = None,
        requirements_file: str | None = None,
        instance_group: list[dict] | None = None,
        test_cases: list[TestCase | dict] | None = None,
    ):
        """A Triton model.

        Args:
            versions: A list of TritonModelVersion objects representing the versions of the model.
            backend: The backend to use for the model, if provided. If not provided, it will be inferred from the model versions.
            config: Inline Config for Triton Model configuration, if provided.
            config_file: The path to the model's Triton configuration file (i.e. models/<model_name>/config.pbtxt), if provided.
            python_version: The python version to use for the model, if provided.
            requirements_file: The path to the model's requirements file (i.e. models/<model_name>/requirements.txt), if provided.
        """
        self.versions = versions or []
        self.versions = [TritonModelVersion(**mv) if isinstance(mv, dict) else mv for mv in self.versions]
        self.python_version = python_version
        self.requirements_file = requirements_file
        self.test_cases = test_cases or []
        self.test_cases = [
            TestCase(**test_case) if isinstance(test_case, dict) else test_case for test_case in self.test_cases
        ]
        self._check_version_numbers_all_or_none()

        self._check_config_and_config_file_isolated(config, config_file)
        if config:
            self.triton_config = parse_pbtxt(content=config) if config else {}
            self.triton_config_text = config if config else ""
        elif config_file:
            self.triton_config = parse_pbtxt(file_path=config_file) if config_file else {}
            self.triton_config_text = Path(config_file).read_text() if config_file else ""
        else:
            self.triton_config = {}
            self.triton_config_text = ""

        self._check_config_dups(self.triton_config, backend, platform, inputs, outputs, max_batch_size, instance_group)
        self.backend = backend or self.triton_config.get("backend", None)
        self.platform = platform or self.triton_config.get("platform", None)
        if inputs:
            inputs = [v if isinstance(v, TritonDType) else TritonDType(**v) for v in inputs]
            self.inputs = [{"name": v.name, "dtype": v.dtype, "dims": v.dims} for v in inputs]
        else:
            self.inputs = self.triton_config.get("input", None)
        if outputs:
            outputs = [v if isinstance(v, TritonDType) else TritonDType(**v) for v in outputs]
            self.outputs = [{"name": v.name, "dtype": v.dtype, "dims": v.dims} for v in outputs]
        else:
            self.outputs = self.triton_config.get("output", None)
        self.max_batch_size = max_batch_size or self.triton_config.get("max_batch_size", None)
        self.instance_group = instance_group or self.triton_config.get("instance_group", None)

        self._check_no_backend_duplicates()

        self.initialized: bool = False
        self.name: str | None = None
        self.path: Path | None = None
        self.flavor: str | None = None
        self.mlflow_model_paths: list[Path] | None = None
        self.mlflow_model_schema: tuple[int, dict, dict] | None = None
        self.ensemble_models: list[tuple[str, int]] | None = None
        self.triton_version: str | None = None
        self.decoupled_transaction_policy: bool | None = None

    @staticmethod
    def _check_config_and_config_file_isolated(config, config_file):
        if config and config_file:
            raise ValueError("Only one of config or config_file should be provided")

    @staticmethod
    def _check_config_dups(
        triton_config: dict,
        backend: str | None,
        platform: str | None,
        inputs: Any | None,
        outputs: Any | None,
        max_batch_size: int | None,
        instance_group: Any | None = None,
    ):
        if "backend" in triton_config and backend:
            raise ValueError("Backend is already specified in the Triton config")
        if "platform" in triton_config and platform:
            raise ValueError("Platform is already specified in the Triton config")
        if inputs and "input" in triton_config:
            raise ValueError("Input is already specified in the Triton config")
        if outputs and "output" in triton_config:
            raise ValueError("Output is already specified in the Triton config")
        if max_batch_size and "max_batch_size" in triton_config:
            raise ValueError("Max batch size is already specified in the Triton config")
        if instance_group and "instance_group" in triton_config:
            raise ValueError("Instance group is already specified in the Triton config")

    def _check_no_backend_duplicates(self):
        if self.backend and self.platform:
            raise ValueError("Both backend and platform are specified, which is not allowed. Please specify only one.")

    def _check_version_numbers_all_or_none(self):
        """Check that all versions have a version number or none of them have a version number"""
        if self.versions:
            version_numbers = [v.version for v in self.versions]
            if not all(version_numbers) and any(version_numbers):
                raise ValueError("All versions must have a version number or none of them should have a version number")

    def init(
        self,
        name: str,
        repo_path: Path,
        triton_version: str,
    ):
        click.secho(f"Initializing {name}")
        self.name = name
        self.path = repo_path.joinpath(self.name)
        self.triton_version = triton_version

        if "name" in self.triton_config and self.triton_config["name"] != self.name:
            raise ValueError(
                f"Model name in Triton config ({self.triton_config['name']}) does not match model name ({self.name})"
            )

        mlflow_artifact_version_uris = []
        for model_version in self.versions:
            if model_version.artifact_uri and model_version.artifact_uri.startswith("models:/"):
                mlflow_artifact_version_uris.append(model_version.artifact_uri)

        self.mlflow_model_paths = list()
        for artifact_uri in mlflow_artifact_version_uris:
            self.mlflow_model_paths.append(download_mlflow_model(artifact_uri))

        mlflow_model_schemas = list()
        for mlflow_model_path in self.mlflow_model_paths:
            with open(Path(mlflow_model_path).joinpath("MLmodel")) as f:
                mlflow_config = yaml.safe_load(f)
            try:
                mlflow_model_schemas.append(parse_mlflow_signature(mlflow_config))
            except (KeyError, ValueError):
                pass

        if mlflow_model_schemas:
            self.mlflow_model_schema = mlflow_model_schemas[0]
            for schema in mlflow_model_schemas[1:]:
                if schema != self.mlflow_model_schema:
                    raise ValueError(
                        f"MLflow model schemas do not match across versions, found: {self.mlflow_model_schema} and {schema}"
                    )

        self.flavor = get_flavor_for_models(mlflow_artifact_version_uris) if mlflow_artifact_version_uris else None
        flavor_backend_map = {  # map mlflow flavors to triton backends
            "python": "mlflow",
            "pytorch": "pytorch",
            "sentence_transformers": "mlflow",
            "transformers": "mlflow",
            "tensorflow": "tensorflow",
            "onnx": "onnxruntime",
        }
        mlflow_implied_backend = flavor_backend_map.get(self.flavor, None)
        if mlflow_artifact_version_uris and mlflow_implied_backend is None:
            raise ValueError(f"Unsupported MLflow flavor: {self.flavor}")

        if not self.backend and not self.platform:
            self.backend = mlflow_implied_backend

        if not self.backend and not self.platform:
            raise ValueError(
                "Backend must be specified or mlflow artifact versions must be specified\n"
                "Please specify a model backend via triton config or by passing the backend directly."
            )

        if self.python_version and self.backend not in {"python", "mlflow"}:
            raise ValueError("python_version can only be specified for python models")

        if not self.python_version and self.backend in {"python", "mlflow"} and mlflow_artifact_version_uris:
            self.python_version = get_python_version_for_mlflow_models(mlflow_artifact_version_uris)
        elif not self.python_version and self.backend in {"python", "mlflow"}:
            self.python_version = python_version_for_triton_version(self.triton_version)

        if self.python_version:
            self.python_version = parse_python_version(self.python_version)

        if self.requirements_file and self.backend not in {"python", "mlflow"}:
            raise ValueError("requirements_file can only be specified for python models")

        if not self.requirements_file and self.backend in {"python", "mlflow"} and mlflow_artifact_version_uris:
            self.requirements_file = get_mlflow_model_requirements(mlflow_artifact_version_uris)

        if self.backend == "pytorch" and self.mlflow_model_schema:
            mlflow_batch_size, mlflow_inputs, mlflow_outputs = self.mlflow_model_schema
            if not self.max_batch_size:
                self.max_batch_size = 1000 if mlflow_batch_size == -1 else mlflow_batch_size
            if not self.inputs:
                self.inputs = [
                    {"name": name, "dtype": dtype, "dims": dims} for name, (dtype, dims, _) in mlflow_inputs.items()
                ]
            if not self.outputs:
                self.outputs = [
                    {"name": name, "dtype": dtype, "dims": dims} for name, (dtype, dims, _) in mlflow_outputs.items()
                ]

        if self.backend == "pytorch" and not self.inputs:
            raise ValueError("inputs must be specified for PyTorch models")

        if self.backend == "pytorch" and not self.outputs:
            raise ValueError("outputs must be specified for PyTorch models")

        self.decoupled_transaction_policy = self.triton_config.get("model_transaction_policy", {}).get(
            "decoupled", False
        )

        default_version = 1
        for model_version in self.versions:
            if model_version.version:
                this_version = model_version.version
            elif model_version.artifact_uri and model_version.artifact_uri.startswith("models:/"):
                _models = find_matching_dbx_models(model_version.artifact_uri)
                if model_version:
                    this_version = int(_models[0].version)
                else:
                    raise ValueError(f"Could not find model version for {model_version.artifact_uri}")
            else:
                this_version = default_version
            version_dir = self.path.joinpath(str(this_version))
            model_version.version = this_version
            model_version.init(
                self.name,
                version_dir,
                self.backend,
                self.platform,
                self.flavor,
                self.test_cases,
                self.decoupled_transaction_policy,
            )
            default_version = this_version + 1

        self.versions = sorted(self.versions, key=lambda mv: mv.version)

        if self.platform == "ensemble":
            if len(self.versions) > 0:
                raise ValueError("Ensemble models cannot have versions specified")
            ensemble_version = TritonModelVersion(version=1)
            ensemble_version.init(
                self.name,
                self.path.joinpath("1"),
                self.backend,
                self.platform,
                self.flavor,
                self.test_cases,
                self.decoupled_transaction_policy,
            )
            self.versions.append(ensemble_version)
            self.ensemble_models = [
                (step["model_name"], step["model_version"]) for step in self.triton_config["ensemble_scheduling"]["step"]
            ]

        self.initialized = True

    def build(self, prep_conda_env: bool = True, prep_triton_stub: bool = True) -> None:
        """
        This method takes the initialized model container and handles processing and structuring the model artifacts.
        """
        self.path.mkdir(parents=True)

        if self.backend == "mlflow":
            assert not self.inputs, "'input' should not be specified in Triton config for MLflow models"
            assert not self.outputs, "'output' should not be specified in Triton config for MLflow models"

            # if max_batch_size is already set, grab it's value and remove it so that it can be re-added later
            if "max_batch_size" in self.triton_config:
                self.max_batch_size = self.triton_config["max_batch_size"]
                # remove max_batch_size from triton_config_text
                lines = self.triton_config_text.splitlines()
                lines = [line for line in lines if "max_batch_size" not in line]
                self.triton_config_text = "\n".join(lines)

            mlflow_triton_config = build_config(
                self.mlflow_model_paths[0], default_max_batch_size=self.max_batch_size or 1000
            )
            self.triton_config_text += "\n" + mlflow_triton_config
            self.triton_config = parse_pbtxt(content=self.triton_config_text)

        if self.backend and "backend" not in self.triton_config:
            self.triton_config_text += f'\nbackend: "{self.backend}"'
            self.triton_config = parse_pbtxt(content=self.triton_config_text)

        if self.platform and "platform" not in self.triton_config:
            self.triton_config_text += f'\nplatform: "{self.platform}"'
            self.triton_config = parse_pbtxt(content=self.triton_config_text)

        if "version_policy" not in self.triton_config:
            self.triton_config_text += (
                f"\nversion_policy: {{ specific: {{ versions: {[mv.version for mv in self.versions]}}}}}"
            )
            self.triton_config = parse_pbtxt(content=self.triton_config_text)

        if "max_batch_size" not in self.triton_config and self.max_batch_size is not None:
            self.triton_config_text += f"\nmax_batch_size: {self.max_batch_size}"
            self.triton_config = parse_pbtxt(content=self.triton_config_text)

        if "instance_group" not in self.triton_config and self.instance_group is not None:
            instance_group_str = "\n".join(
                [
                    "\t{\n"
                    + "\n".join([f'\t\t{k}: {v if isinstance(v, int) else f"""{v}"""}' for k, v in ig.items()])
                    + "\n\t}"
                    for ig in self.instance_group
                ]
            )
            self.triton_config_text += f"\ninstance_group: [\n{instance_group_str}\n]"
            self.triton_config = parse_pbtxt(content=self.triton_config_text)

        if "input" not in self.triton_config and self.inputs:
            for triton_input in self.inputs:
                name, dtype, dims = triton_input["name"], triton_input["dtype"], triton_input["dims"]
                self.triton_config_text += (
                    f'\ninput [\n\t{{\n\t\tname: "{name}"\n\t\tdata_type: TYPE_{dtype}\n\t\tdims: {dims}\n\t}}\n]'
                )
            self.triton_config = parse_pbtxt(content=self.triton_config_text)

        if "output" not in self.triton_config and self.outputs:
            for triton_output in self.outputs:
                name, dtype, dims = triton_output["name"], triton_output["dtype"], triton_output["dims"]
                self.triton_config_text += (
                    f'\noutput [\n\t{{\n\t\tname: "{name}"\n\t\tdata_type: TYPE_{dtype}\n\t\tdims: {dims}\n\t}}\n]'
                )
            self.triton_config = parse_pbtxt(content=self.triton_config_text)

        if self.requirements_file and prep_conda_env:
            conda_pack_path, _, conda_pack_info = build_conda_env(
                self.python_version,
                Path(self.requirements_file),
            )
            dst_path = self.path.joinpath(conda_pack_path.name)
            click.secho(f"Creating hard link: {dst_path} -> {conda_pack_path}", fg="green")
            # rel_path = os.path.relpath(conda_pack_path, dst_path.parent)
            link_or_copy(conda_pack_path, dst_path)

            # point to python execution env, if it exists
            self.triton_config_text += f'\nparameters: {{key: "EXECUTION_ENV_PATH" value: {{string_value: "$$TRITON_MODEL_DIRECTORY/{conda_pack_path.name}"}}}}\n'

        if (
            self.python_version
            and self.python_version != python_version_for_triton_version(self.triton_version)
            and prep_triton_stub
        ):
            stub_path = build_triton_stub(self.python_version, f"r{self.triton_version}")
            dst_path = self.path.joinpath("triton_python_backend_stub")
            click.secho(f"Copying Triton Stub: {dst_path} -> {stub_path}", fg="green")
            shutil.copy2(stub_path, dst_path)

        for model_version in self.versions:
            click.secho(f"processing version: {model_version.version}", fg="green")
            model_version.build()

        self.path.joinpath("config.pbtxt").write_text(self.triton_config_text)

    def create_test_plan(self) -> TritonModelTestPlan:
        """Create a test plan for this model version."""
        if not self.initialized:
            raise ValueError("Model version must be initialized before creating a test plan")
        return TritonModelTestPlan(
            version_plans=[version.create_test_plan() for version in self.versions], ensemble_models=self.ensemble_models
        )
