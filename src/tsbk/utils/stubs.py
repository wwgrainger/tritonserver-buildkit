import platform
from pathlib import Path

import click
from mlflow_backend_utils.sdk import build_triton_stub as _build_triton_stub

from tsbk import TSBK_DIR, TSBK_K8S_SERVICE_ACCOUNT, TSBK_S3_PREFIX
from tsbk.utils.cache import append_cache_bust


def build_triton_stub(
    python_version: str,
    triton_version: str,
    cache_bust: str | None = None,
) -> Path:
    """Builds a conda environment given the requirements and packages it up with conda-pack and writes
    the resulting archive to the output_dir

    Args:
        python_version: the python version to use for the environment
        triton_version: the version of Triton to use for the stub
        cache_bust: optional value used to invalidate the cached stub

    Returns:
        path to the output stub
    """
    arch = platform.machine()

    output_dir = TSBK_DIR.joinpath("triton_python_stubs")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_key = append_cache_bust(f"triton_python_backend_stub-{triton_version}-{arch}-{python_version}", cache_bust)
    output_path = output_dir.joinpath(cache_key)

    if output_path.exists():
        return output_path

    if TSBK_S3_PREFIX:
        s3_path = f"{TSBK_S3_PREFIX}/triton_python_stubs/{cache_key}"
    else:
        s3_path = None

    click.secho(f"Building Triton Python Stub for {triton_version}/{python_version} on {arch}", fg="blue")
    _build_triton_stub(
        python_version=python_version,
        triton_version=triton_version,
        platform=arch,
        output_path=output_path,
        k8s_shared_s3_path=s3_path,
        k8s_service_account=TSBK_K8S_SERVICE_ACCOUNT,
    )

    return output_path
