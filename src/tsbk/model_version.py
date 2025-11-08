import shutil
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from tsbk.test_case import TestCase
from tsbk.testing import TritonModelVersionTestPlan
from tsbk.utils import link_or_copy
from tsbk.utils.dbx import download_mlflow_model, get_input_example_from_model
from tsbk.utils.s3 import download_s3_path, s3_path_exists


class TritonModelVersion:
    def __init__(
        self,
        artifact_uri: str | None = None,
        python_model_file: str | None = None,
        version: int | None = None,
        test_cases: list[TestCase | dict] | None = None,
    ):
        """A Triton model version.

        Args:
            artifact_uri: The URI of the model artifact, which can be an MLflow model or an S3 object.
            python_model_file: The path to the Python model file, which is required for Python models.
            version: The version number of the model.
        """
        self.artifact_uri = artifact_uri
        self.python_model_file = python_model_file
        self.version = version
        self.test_cases = test_cases or []
        self.test_cases = [
            TestCase(**test_case) if isinstance(test_case, dict) else test_case for test_case in self.test_cases
        ]
        self._check_version_number()

        self.initialized: bool = False
        self.name: str | None = None
        self.path: Path | None = None
        self.backend: str | None = None
        self.platform: str | None = None
        self.flavor: str | None = None
        self.decoupled_transaction_policy: bool | None = None

    def __str__(self):
        return f"TritonModelVersion(artifact_uri={self.artifact_uri}, python_model_file={self.python_model_file}, version={self.version})"

    def _check_version_number(self):
        """Check that version is greater than 0"""
        if self.version is not None and self.version < 1:
            raise ValueError("version must be greater than 0")

    def init(
        self,
        name: str,
        path: Path,
        backend: str | None,
        platform: str | None,
        flavor: str | None,
        model_tests: list[TestCase],
        decoupled_transaction_policy: bool,
    ) -> None:
        """Initialize the model version with the given parameters."""
        self.name = name
        self.path = path
        self.backend = backend
        self.platform = platform
        self.flavor = flavor
        self.decoupled_transaction_policy = decoupled_transaction_policy
        for test in model_tests:
            self.test_cases.append(deepcopy(test))

        if self.artifact_uri and self.artifact_uri.startswith("models:/"):
            input_example = get_input_example_from_model(self.artifact_uri)
            if input_example is not None:
                self.test_cases.append(TestCase(inputs=input_example, expected_outputs={}))

        self._check_backend_or_platform()
        self._check_artifact_uri_or_python_model_file()
        self._check_python_model_file()
        self._check_artifact_uri()
        self.initialized = True

    def _check_backend_or_platform(self):
        if self.backend is None and self.platform is None:
            raise ValueError("Either backend or platform must be specified")
        if self.backend is not None and self.platform is not None:
            raise ValueError("Only one of backend or platform can be specified")

    def _check_artifact_uri_or_python_model_file(self):
        """Check that either artifact_uri or python_model_file is provided"""
        if self.platform != "ensemble" and (self.artifact_uri is None and self.python_model_file is None):
            raise ValueError("Either artifact_uri or python_model_file must be provided")

    def _check_python_model_file(self):
        if self.backend == "python" and not self.python_model_file and self.artifact_source[0] != "mlflow":
            raise ValueError("python_model_file must be specified for Python models that are not MLflow models")

    def _check_artifact_uri(self):
        if self.artifact_uri:
            uri = urlparse(self.artifact_uri)
            if uri.scheme not in ["models", "s3"]:
                raise ValueError(f"artifact_uri must be either an MLflow or S3 object, got {uri.scheme}")
            if uri.scheme == "s3" and not s3_path_exists(self.artifact_uri):
                raise ValueError(f"artifact_uri {self.artifact_uri} does not exist")

    def build(self) -> None:
        """Takes a configuration block defining a specific model version and handles the copying and organization of model assets in the version directory in a Triton-compatible format."""
        self.path.mkdir(parents=True)
        if self.artifact_uri:
            source, func = self.artifact_source
            copy_func = partial(func, origin_path=self.artifact_uri)

            match self.backend:
                case "mlflow":
                    copy_func(dst_path=self.path)
                case "python":
                    if source == "mlflow":
                        output_file_path = self.path.joinpath("mlflow_model").as_posix()
                        copy_func(dst_path=output_file_path)
                    else:
                        output_file_path = self.path.joinpath(self.artifact_uri.split("/")[-1])
                        copy_func(dst_path=output_file_path)

                case "pytorch":
                    if source == "mlflow":
                        model_path = copy_func()
                        # rel_path = os.path.relpath(model_path.joinpath("data/model.pth"), self.path)
                        link_or_copy(
                            model_path.joinpath("data/model.pth"),
                            self.path.joinpath("model.pt"),
                        )
                    else:
                        output_file_path = self.path.joinpath("model.pt").as_posix()
                        copy_func(dst_path=output_file_path)

                case "onnxruntime":
                    if source == "mlflow":
                        self.path.rmdir()
                        copy_func(dst_path=self.path)
                    else:
                        output_file_path = self.path.joinpath("model.onnx").as_posix()
                        copy_func(dst_path=output_file_path)

                case "tensorrt":
                    output_file_path = self.path.joinpath("model.plan").as_posix()
                    copy_func(dst_path=output_file_path)

                case _:
                    output_file_path = self.path.joinpath(self.artifact_uri.split("/")[-1])
                    copy_func(dst_path=output_file_path)

        if self.python_model_file:
            # Note: for now we expect that python_model_file to be built into container at WORKING_DIR/models/ path
            shutil.copy(
                self.python_model_file,
                self.path.joinpath("model.py"),
            )

        # ensemble models have no version assets, but need a dummy model version folder to be recognized by Triton
        if self.platform == "ensemble":
            keep_path = self.path.joinpath(".keep")
            keep_path.parent.mkdir(exist_ok=True)
            with open(keep_path, "w") as fp:
                fp.write("Need this model version folder for ensemble model.")

    @property
    def artifact_source(self) -> tuple[Optional[str], Optional[callable]]:
        """
        Determines the source of a given artifact_uri (S3 or MLFlow) and provides the appropriate function to copy the artifact from source to version folder.

        Returns:
            source: "mlflow" or "S3"
            copy_function: Function to use to copy the artifact from its origin to the Triton model version folder.
            profile: Profile to use if artifact is a DBX object. If the artifact is an S3 object, this is None
        """
        if not self.artifact_uri:
            return None, None
        elif self.artifact_uri.startswith("models:/"):
            return "mlflow", download_mlflow_model
        elif self.artifact_uri.startswith("s3://"):
            return "s3", download_s3_path
        raise ValueError(f"Invalid artifact_uri: {self.artifact_uri}")

    def create_test_plan(self) -> TritonModelVersionTestPlan:
        """Create a test plan for this model version."""
        if not self.initialized:
            raise ValueError("Model version must be initialized before creating a test plan")
        return TritonModelVersionTestPlan(
            name=self.name,
            version=self.version,
            test_cases=self.test_cases,
            decoupled_transaction_policy=self.decoupled_transaction_policy,
        )
