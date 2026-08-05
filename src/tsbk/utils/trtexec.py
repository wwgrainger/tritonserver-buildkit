import hashlib
import os
import platform
import random
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Literal

import boto3
import click
import yaml

from tsbk import (
    DEFAULT_TRITON_VERSION,
    TSBK_DIR,
    TSBK_K8S_SERVICE_ACCOUNT,
    TSBK_S3_PREFIX,
)
from tsbk.utils.cache import cache_bust_key_material

DEFAULT_TRT_IMAGE = f"nvcr.io/nvidia/tensorrt:{DEFAULT_TRITON_VERSION}-py3"


def build_trt_engine(
    onnx_path: Path,
    precision: str | None = None,
    workspace_size: int | None = None,
    extra_args: str | None = None,
    trt_image: str | None = None,
    gpu_name: str | None = None,
    instance_family: str | None = None,
    preferred_methods: Iterable[Literal["docker", "kubernetes"]] = ("docker", "kubernetes"),
    cache_bust: str | None = None,
) -> Path:
    """Compiles an ONNX model to a TensorRT engine plan file.

    Args:
        onnx_path: Path to the ONNX model file
        precision: Precision mode ('fp16', 'int8', 'best', or None for default)
        workspace_size: Max workspace size in MB for trtexec
        extra_args: Additional raw trtexec CLI arguments
        trt_image: Override TensorRT container image
        gpu_name: Target GPU architecture for K8s scheduling (e.g., 'A10G', 'T4')
        instance_family: AWS instance family for K8s scheduling via karpenter (e.g., 'g5', 'p4d')
        preferred_methods: Execution methods to try in order
        cache_bust: optional value used to invalidate the cached engine

    Returns:
        Path to the compiled .plan file
    """
    onnx_path = Path(onnx_path).resolve()
    trt_image = trt_image or DEFAULT_TRT_IMAGE
    arch = platform.machine()

    # Build cache key from ONNX content hash + compile params
    onnx_hash = hashlib.sha256(onnx_path.read_bytes()).hexdigest()[:16]
    params_str = f"{precision or 'default'}-{workspace_size or 'default'}-{extra_args or ''}-{gpu_name or 'any'}-{instance_family or 'any'}-{arch}"
    params_hash = hashlib.sha256(params_str.encode() + cache_bust_key_material(cache_bust)).hexdigest()[:8]
    cache_key = f"{onnx_hash}-{params_hash}"

    output_dir = TSBK_DIR.joinpath("trt_engines")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir.joinpath(f"{cache_key}.plan")

    if output_path.exists():
        click.secho(f"TRT engine cache hit: {output_path}", fg="green")
        return output_path

    if TSBK_S3_PREFIX:
        s3_path = f"{TSBK_S3_PREFIX}/trt_engines/{cache_key}.plan"
    else:
        s3_path = None

    click.secho(
        f"Compiling ONNX to TensorRT engine: {onnx_path.name} " f"(precision={precision or 'default'}, arch={arch})",
        fg="blue",
    )

    method_maps = {
        "docker": (_can_compile_docker, _compile_docker),
        "kubernetes": (_can_compile_kubernetes, _compile_kubernetes),
    }

    for method in preferred_methods:
        if method not in method_maps:
            raise ValueError(f"Unknown method: {method}. Supported methods are: {list(method_maps.keys())}")
        can_build_fn, build_fn = method_maps[method]
        can_build, reason = can_build_fn(shared_s3_path=s3_path, gpu_name=gpu_name)
        if can_build:
            build_fn(
                onnx_path=onnx_path,
                output_path=output_path,
                precision=precision,
                workspace_size=workspace_size,
                extra_args=extra_args,
                trt_image=trt_image,
                gpu_name=gpu_name,
                instance_family=instance_family,
                k8s_shared_s3_path=s3_path,
                k8s_service_account=TSBK_K8S_SERVICE_ACCOUNT,
            )
            return output_path
        else:
            click.secho(f"Cannot compile ONNX to TRT with {method}: {reason}", fg="yellow")

    raise RuntimeError(
        f"Cannot compile ONNX to TensorRT in current environment with methods: {list(preferred_methods)}. "
        f"trtexec requires GPU access via Docker (--gpus all) or a Kubernetes cluster with GPU nodes."
    )


def _build_trtexec_command(
    onnx_input: str,
    plan_output: str,
    precision: str | None = None,
    workspace_size: int | None = None,
    extra_args: str | None = None,
) -> str:
    """Constructs a trtexec command string.

    Args:
        onnx_input: Path to the ONNX model inside the container
        plan_output: Path for the output .plan file inside the container
        precision: Precision mode ('fp16', 'int8', 'best', or None)
        workspace_size: Max workspace size in MB
        extra_args: Additional raw trtexec arguments

    Returns:
        The trtexec command string
    """
    cmd = f"trtexec --onnx={onnx_input} --saveEngine={plan_output}"
    if precision == "fp16":
        cmd += " --fp16"
    elif precision == "int8":
        cmd += " --int8"
    elif precision == "best":
        cmd += " --best"
    elif precision is not None:
        raise ValueError(f"Unknown precision mode: {precision}. Supported: fp16, int8, best")
    if workspace_size is not None:
        cmd += f" --workspace={workspace_size}"
    if extra_args:
        cmd += f" {extra_args}"
    return cmd


# --- Docker execution ---


def _can_compile_docker(shared_s3_path: str | None = None, gpu_name: str | None = None, **_) -> tuple[bool, str]:
    """Check if Docker with GPU support is available for TRT compilation."""
    if shutil.which("docker") is None:
        return False, "docker command not found"
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False, f"docker info failed: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, "docker info timed out"

    # Check docker has the NVIDIA runtime (nvidia-container-toolkit)
    if "nvidia" not in result.stdout.lower():
        return False, "Docker does not have the NVIDIA runtime configured (install nvidia-container-toolkit)"

    # Check nvidia-smi for GPU availability
    if shutil.which("nvidia-smi") is None:
        return False, "nvidia-smi not found; no NVIDIA GPU driver detected"
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if smi.returncode != 0:
            return False, f"nvidia-smi failed: {smi.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi timed out"

    detected_gpus = [line.strip() for line in smi.stdout.strip().splitlines() if line.strip()]
    if not detected_gpus:
        return False, "nvidia-smi reported no GPUs"

    # If a specific GPU was requested, verify it is present
    if gpu_name and not any(gpu_name.lower() in gpu.lower() for gpu in detected_gpus):
        return False, f"requested GPU '{gpu_name}' not found; detected: {', '.join(detected_gpus)}"

    return True, f"Docker with GPU available (detected: {', '.join(detected_gpus)})"


def _compile_docker(
    *,
    onnx_path: Path,
    output_path: Path,
    precision: str | None = None,
    workspace_size: int | None = None,
    extra_args: str | None = None,
    trt_image: str = DEFAULT_TRT_IMAGE,
    **_,
) -> None:
    """Compile ONNX to TensorRT using Docker with GPU access."""
    onnx_path = Path(onnx_path).resolve()
    output_path = Path(output_path)

    trtexec_cmd = _build_trtexec_command(
        onnx_input="/workspace/model.onnx",
        plan_output="/workspace/model.plan",
        precision=precision,
        workspace_size=workspace_size,
        extra_args=extra_args,
    )

    volumes = [
        "-v",
        f"{onnx_path.parent}:/workspace",
    ]

    env = []
    if "REQUESTS_CA_BUNDLE" in os.environ:
        container_cert_path = "/tmp/requests_ca_bundle.pem"
        volumes.extend(["-v", f"{os.environ['REQUESTS_CA_BUNDLE']}:{container_cert_path}"])
        env.extend(["-e", f"REQUESTS_CA_BUNDLE={container_cert_path}"])

    result = subprocess.run(
        args=[
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            *volumes,
            *env,
            trt_image,
            "bash",
            "-c",
            f"cd /workspace && {trtexec_cmd}",
        ],
    )
    if result.returncode != 0:
        raise RuntimeError("TensorRT engine compilation failed in Docker")

    # Move plan from ONNX directory to cache location
    plan_in_workspace = onnx_path.parent / "model.plan"
    shutil.move(str(plan_in_workspace), str(output_path))


# --- Kubernetes execution ---


def _can_compile_kubernetes(shared_s3_path: str | None = None, **_) -> tuple[bool, str]:
    """Check if Kubernetes is available for TRT compilation."""
    if shutil.which("kubectl") is None:
        return False, "kubectl command not found"
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return False, f"Unable to connect to Kubernetes cluster: {e}"
    if shared_s3_path is None:
        return False, "TSBK_S3_PREFIX is required for Kubernetes TRT compilation (set env var TSBK_S3_PREFIX)"
    return True, "Kubernetes is available"


def _compile_kubernetes(
    *,
    onnx_path: Path,
    output_path: Path,
    precision: str | None = None,
    workspace_size: int | None = None,
    extra_args: str | None = None,
    trt_image: str = DEFAULT_TRT_IMAGE,
    gpu_name: str | None = None,
    instance_family: str | None = None,
    k8s_shared_s3_path: str,
    k8s_service_account: str = "default",
    **_,
) -> None:
    """Compile ONNX to TensorRT using a Kubernetes Job with GPU."""
    onnx_path = Path(onnx_path).resolve()
    output_path = Path(output_path)

    bucket_name, plan_key = k8s_shared_s3_path.replace("s3://", "").split("/", 1)

    # Upload ONNX model and any external data files (.data) to an S3 directory
    model_dir_key = plan_key.replace(".plan", "_model/")
    s3_model_dir = f"s3://{bucket_name}/{model_dir_key}"

    model_files = [onnx_path] + sorted(onnx_path.parent.glob("*.data"))

    s3 = boto3.client("s3")
    uploaded_keys = []
    click.secho(f"Uploading {len(model_files)} model file(s) to S3: {s3_model_dir}", fg="blue")
    for f in model_files:
        key = f"{model_dir_key}{f.name}"
        s3.upload_file(str(f), bucket_name, key)
        uploaded_keys.append(key)

    # Create and run K8s job
    job_name = f"tsbk-trt-compile-{random.randint(0, 99999):05d}"
    job_manifest = _create_trt_job_manifest(
        job_name=job_name,
        s3_model_dir=s3_model_dir,
        s3_plan_path=k8s_shared_s3_path,
        onnx_filename=onnx_path.name,
        precision=precision,
        workspace_size=workspace_size,
        extra_args=extra_args,
        trt_image=trt_image,
        gpu_name=gpu_name,
        instance_family=instance_family,
        service_account=k8s_service_account,
    )

    _remove_job_if_exists(job_name)
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=yaml.dump(job_manifest),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create Kubernetes job: {result.stderr}")

    click.secho(f"K8s TRT compile job created: {job_name}", fg="blue")

    # Wait for job to start running
    result = subprocess.run(
        [
            "kubectl",
            "wait",
            "--for=jsonpath={.status.ready}=1",
            "--timeout=600s",
            f"job/{job_name}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to wait for Kubernetes job to start: {result.stderr}")

    # Follow logs
    subprocess.run(["kubectl", "logs", "-f", f"job/{job_name}"])

    # Wait for completion
    for _ in range(60 * 10 // 10):
        result = subprocess.run(
            ["kubectl", "wait", "--for=condition=complete", "--timeout=10s", f"job/{job_name}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            break

        job_status = subprocess.run(
            ["kubectl", "get", "job", job_name, "-o", "jsonpath={.status.failed}"],
            capture_output=True,
            text=True,
            check=True,
        )
        if job_status.stdout.strip() not in ("", "0"):
            raise RuntimeError(f"Kubernetes TRT compile job {job_name} has failed. Check the logs for details.")
    else:
        raise RuntimeError(f"Kubernetes TRT compile job {job_name} timed out")

    # Download plan from S3
    click.secho(f"Downloading TRT engine from S3: {k8s_shared_s3_path}", fg="blue")
    try:
        s3.download_file(bucket_name, plan_key, str(output_path))
    except Exception as e:
        raise RuntimeError(f"Failed to download TRT engine from S3: {e}")

    # Cleanup model files from S3
    for key in uploaded_keys:
        try:
            s3.delete_object(Bucket=bucket_name, Key=key)
        except Exception:
            pass


def _create_trt_job_manifest(
    job_name: str,
    s3_model_dir: str,
    s3_plan_path: str,
    onnx_filename: str = "model.onnx",
    precision: str | None = None,
    workspace_size: int | None = None,
    extra_args: str | None = None,
    trt_image: str = DEFAULT_TRT_IMAGE,
    gpu_name: str | None = None,
    instance_family: str | None = None,
    cpu: str = "2",
    memory: str = "8Gi",
    memory_limit: str = "16Gi",
    service_account: str = "default",
) -> dict:
    """Create a Kubernetes Job manifest for TRT compilation."""
    trtexec_cmd = _build_trtexec_command(
        onnx_input=f"/workspace/{onnx_filename}",
        plan_output="/workspace/model.plan",
        precision=precision,
        workspace_size=workspace_size,
        extra_args=extra_args,
    )

    # --recursive downloads the .onnx and any external .data files
    s3_model_dir = s3_model_dir.rstrip("/") + "/"
    compile_script = f"""set -ex
pip install awscli 2>/dev/null || true
aws s3 cp --recursive {s3_model_dir} /workspace/
{trtexec_cmd}
aws s3 cp /workspace/model.plan {s3_plan_path}
"""

    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "annotations": {"karpenter.sh/do-not-disrupt": "true"},
        },
        "spec": {
            "ttlSecondsAfterFinished": 60 * 60,
            "backoffLimit": 0,
            "template": {
                "metadata": {
                    "labels": {
                        "tsbk-job-id": job_name,
                    },
                },
                "spec": {
                    "serviceAccountName": service_account,
                    "restartPolicy": "Never",
                    "tolerations": [
                        {
                            "effect": "NoSchedule",
                            "key": "gpu",
                            "value": "true",
                        },
                    ],
                    "containers": [
                        {
                            "name": "trtexec",
                            "image": trt_image,
                            "imagePullPolicy": "Always",
                            "command": ["/bin/sh", "-c", compile_script],
                            "resources": {
                                "requests": {
                                    "cpu": cpu,
                                    "memory": memory,
                                    "nvidia.com/gpu": "1",
                                },
                                "limits": {
                                    "memory": memory_limit,
                                    "nvidia.com/gpu": "1",
                                },
                            },
                            "volumeMounts": [],
                            "env": [],
                            "envFrom": [],
                        }
                    ],
                    "volumes": [],
                },
            },
        },
    }

    node_selector = {}
    if gpu_name:
        node_selector["karpenter.k8s.aws/instance-gpu-name"] = gpu_name.lower()
    if instance_family:
        node_selector["karpenter.k8s.aws/instance-family"] = instance_family.lower()
    if node_selector:
        job_manifest["spec"]["template"]["spec"]["nodeSelector"] = node_selector

    return job_manifest


def _remove_job_if_exists(job_name: str) -> None:
    """Remove a Kubernetes job if it already exists."""
    subprocess.run(
        ["kubectl", "delete", "job", job_name],
        capture_output=True,
        text=True,
        check=False,
    )


def _find_onnx_file(version_path: Path) -> Path:
    """Find the ONNX model file in a version directory.

    Args:
        version_path: Path to the model version directory

    Returns:
        Path to the ONNX model file

    Raises:
        RuntimeError: If no ONNX file is found
    """
    # Check flat directory first
    onnx_files = list(version_path.glob("*.onnx"))
    if len(onnx_files) == 1:
        return onnx_files[0]
    if len(onnx_files) > 1:
        return onnx_files[0]

    # For MLflow-sourced models, check subdirectories
    onnx_files = list(version_path.rglob("*.onnx"))
    if onnx_files:
        return onnx_files[0]

    raise RuntimeError(f"Cannot find ONNX model file in {version_path} for TRT compilation")
