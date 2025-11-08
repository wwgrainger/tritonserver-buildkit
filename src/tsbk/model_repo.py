import os.path
import shutil
import subprocess
import threading
import time
from pathlib import Path

import click
import mlflow_backend
import requests

from tsbk import DEFAULT_TRITON_VERSION, TSBK_DIR
from tsbk.model import TritonModel
from tsbk.test_case import TestResult
from tsbk.testing import TritonModelRepoTestPlan


class TritonModelRepo:
    def __init__(
        self,
        name: str,
        path: str | Path,
        models: dict[str, TritonModel | dict],
        triton_image: str = "nvcr.io/nvidia/tritonserver",
        triton_image_tag: str = f"{DEFAULT_TRITON_VERSION}-py3",
    ):
        """A Triton Model Repository.

        Args:
            name: The name of the model repository, which is used as the directory name in the Triton model repository.
            models: A dict of Triton models to deploy in the model repository.
            triton_image: The Docker image to use for the Triton server.
            triton_image_tag: The Docker image tag to use for the Triton server.
        """
        self.name = name
        self.models = models
        self.triton_image = triton_image
        self.triton_image_tag = triton_image_tag
        self.triton_version = self.parse_triton_version(self.triton_image_tag)

        self.models = {
            name: TritonModel(**model) if isinstance(model, dict) else model for name, model in self.models.items()
        }

        self.path = Path(path).absolute()
        for name, model in self.models.items():
            model.init(name=name, repo_path=self.path, triton_version=self.triton_version)

        self._resolve_ensemble_models()

        self.container_id = None
        self.http_url = None
        self.grpc_url = None
        self.metrics_url = None
        self.log_thread = None
        self.mlflow_backend_path = None

    @staticmethod
    def parse_triton_version(triton_image_tag: str):
        triton_version = triton_image_tag.split("-")[0] if "-" in triton_image_tag else triton_image_tag
        try:
            major, minor = triton_version.split(".")
            _ = int(major), int(minor)
            return f"{major}.{minor}"
        except ValueError:
            raise ValueError(
                f"Could not parse Triton version from image tag: {triton_image_tag}. Expected format is 'major.minor'"
            )

    def _resolve_ensemble_models(self):
        """Resolves ensemble models by ensuring all required models are included in the repo."""
        for model_name, model in self.models.items():
            if model.ensemble_models:
                for i, (ens_model_name, ens_model_version) in enumerate(model.ensemble_models):
                    if not any(_m == ens_model_name for _m in self.models.keys()):
                        raise ValueError(
                            f"Ensemble model {ens_model_name}:{ens_model_version} not found in the model repository."
                        )
                    if ens_model_version == -1:
                        self.models[model_name].ensemble_models[i] = (
                            ens_model_name,
                            self.models[ens_model_name].versions[-1].version,
                        )
                    elif not any(_mv.version == ens_model_version for _mv in self.models[ens_model_name].versions):
                        raise ValueError(
                            f"Ensemble model {ens_model_name}:{ens_model_version} not found in the model repository."
                        )

        click.secho("All ensemble models resolved successfully.", fg="green")

    @property
    def uses_mlflow_backend(self):
        return any([model.backend == "mlflow" for model in self.models.values()])

    def build(self):
        """Prepares all the models in this repo for serving."""
        if self.path.exists():
            click.secho("Warning: model repo already exists and will be overwritten", fg="yellow")
        shutil.rmtree(self.path, ignore_errors=True)
        click.secho("beginning model preparation...", fg="blue")
        for n, model in enumerate(self.models.values()):
            click.secho(f"preparing model {n+1} of {len(self.models)}: {model.name}", fg="blue")
            model.build()

        if self.uses_mlflow_backend:
            backend_dir = Path(mlflow_backend.__file__).parent.absolute().as_posix()
            backend_version = mlflow_backend.__version__
            self.mlflow_backend_path = TSBK_DIR.joinpath("backends", f"mlflow-{backend_version}")
            if not self.mlflow_backend_path.exists():
                click.secho(f"Copying mlflow backend to {self.mlflow_backend_path}", fg="blue")
                shutil.copytree(backend_dir, self.mlflow_backend_path, dirs_exist_ok=True)

    def run(
        self,
        http_port: int = 8000,
        grpc_port: int = 8001,
        metrics_port: int = 8002,
        shm_size: str = "4G",
        gpus: bool = False,
        env_file: str | None = None,
        detach: bool = False,
    ):
        click.secho("Running tritonserver in container...", fg="blue")
        http_port = http_port or ""
        grpc_port = grpc_port or ""
        metrics_port = metrics_port or ""

        mount_path = os.path.commonpath([TSBK_DIR.resolve(), self.path.resolve()])
        cmd = [
            "docker",
            "run",
            "-v",
            f"{mount_path}:/data",
            "-e",
            "MLFLOW_TRACKING_URI=file:///root/blackhole/mlruns",
            "--shm-size",
            shm_size,
        ]
        if self.uses_mlflow_backend:
            cmd.extend(["-v", f"{self.mlflow_backend_path.absolute().as_posix()}:/opt/tritonserver/backends/mlflow"])

        for host_port, target_port in [(http_port, 8000), (grpc_port, 8001), (metrics_port, 8002)]:
            cmd.extend([f"-p{host_port}:{target_port}"])

        if gpus:
            cmd.extend(["--gpus", "all"])

        if env_file:
            cmd.extend(["--env-file", env_file])

        if detach:
            cmd.append("-d")

        cmd.extend([f"{self.triton_image}:{self.triton_image_tag}"])

        cmd.extend(
            [
                f"tritonserver",
                f"--model-repository=/data/{self.path.relative_to(mount_path)}",
                "--strict-readiness",
                "true",
            ]
        )

        click.secho(f"Running command:\n{' '.join(cmd)}", fg="blue")

        result = subprocess.run(cmd, check=False, capture_output=detach)
        if result.returncode != 0:
            if detach:
                click.secho(f"Error running tritonserver: {result.stderr.decode()}", fg="red")
            raise RuntimeError("Error running tritonserver")

        if detach:
            click.secho("Triton server is running in detached mode.", fg="green")
            self.container_id = result.stdout.decode().strip()
            self._follow_container_logs()
            self.http_url = f"http://localhost:{http_port}"
            self.grpc_url = f"http://localhost:{grpc_port}"
            self.metrics_url = f"http://localhost:{metrics_port}/metrics"
            try:
                self._wait_for_server_ready(self.http_url)
            except KeyboardInterrupt:
                click.secho("Interrupted while waiting for server to be ready. Stopping container...", fg="red")
                self.stop()
                raise

    def _wait_for_server_ready(self, url: str, timeout: int = 60 * 5):
        """Waits for the Triton server to be ready by checking the health endpoint."""
        start_time = time.time()
        while True:
            try:
                response = requests.get(f"{url}/v2/health/ready")
                if response.status_code == 200:
                    click.secho("Triton server is ready!", fg="green")
                    return
            except requests.RequestException as e:
                click.secho(f"Waiting for Triton server to be ready: {e}", fg="yellow")
            if not self._container_still_running():
                raise RuntimeError("Triton server container has stopped unexpectedly.")
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Triton server did not become ready within {timeout} seconds.")
            time.sleep(5)

    def _container_still_running(self):
        """Checks if the Triton server container is still running."""
        if not self.container_id:
            return False
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", self.container_id],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() == "true"
        except subprocess.CalledProcessError:
            return False

    def _follow_container_logs(self):
        """Starts a new subprocess to follow and print container logs."""
        if not self.container_id:
            click.secho("No Triton server container is currently running.", fg="yellow")
            return

        def follow_logs():
            try:
                subprocess.run(["docker", "logs", "-f", self.container_id], check=True)
            except subprocess.CalledProcessError:
                click.secho("Error following container logs.", fg="red")

        self.log_thread = threading.Thread(target=follow_logs, daemon=True)
        self.log_thread.start()

    def test(
        self, url: str, ca_certs: str | None = None, headers: dict | None = None, grpc: bool = False
    ) -> list[TestResult]:
        """Runs the test cases defined for all models in this repository against the Triton server.

        Args:
            url: The URL of the Triton server to test against.
            ca_certs: Path to CA certificates for secure connections, if applicable.
            headers: Additional headers to include in the requests.
            grpc: Whether to use gRPC for testing.

        Returns:
            A list of TestResult objects containing the results of the tests.
        """
        test_plan = self.create_test_plan()
        return test_plan.run_tests(url=url, ca_certs=ca_certs, headers=headers, grpc=grpc)

    def create_test_plan(self) -> TritonModelRepoTestPlan:
        return TritonModelRepoTestPlan(
            model_plans=[model.create_test_plan() for model in self.models.values()],
            repo_models=self.repo_models,
        )

    @property
    def repo_models(self) -> list[tuple[str, int]]:
        """Returns a list of tuples of model name and version for all models in the repo."""
        repo_models = list()
        for model in self.models.values():
            for version in model.versions:
                repo_models.append((model.name, version.version))
        return repo_models

    def stop(self):
        """Stops the Triton server container if it is running."""
        if self.container_id:
            click.secho(f"Stopping Triton server container {self.container_id}...", fg="blue")
            subprocess.run(["docker", "stop", self.container_id], check=True)
            subprocess.run(["docker", "rm", self.container_id], check=True)
            self.container_id = None
        else:
            click.secho("No Triton server container is currently running.", fg="yellow")
