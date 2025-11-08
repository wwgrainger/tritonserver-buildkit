import hashlib
import json
from pathlib import Path

import click
from mlflow_backend_utils.sdk import build_conda_pack as _build_conda_pack

from tsbk import TSBK_DIR
from tsbk.utils import get_platform_desc

CONDA_BUILD_VERSION = 5


def calc_hash(
    requirements_file_bytes: bytes,
    platform_bytes: bytes,
    python_version: str,
    build_env: dict[str, str] = None,
    conda_extras: str = "",
) -> str:
    """Calculates the hash for the environment based on requirements and platform

    Args:
        requirements_file_bytes: byte representation of requirements file contents i.e. Path(path/to/requirements.txt).read_bytes()
        platform_bytes: bytes describing current platform (from import platform), platform.encode()
        python_version: python version for environment
        build_env: extra environment variables to set when running the conda build script
        conda_extras: extra conda packages to install

    Returns:
        hash containing conda env information used as file name for this env in model repo and s3 cache.
    """
    hash_str = (
        requirements_file_bytes
        + platform_bytes
        + python_version.encode()
        + str(CONDA_BUILD_VERSION).encode()
        + conda_extras.encode()
    )

    if build_env:
        hash_str += "".join([f"{k}_{v}" for k, v in sorted(build_env.items())]).encode()

    return hashlib.sha256(hash_str).hexdigest()


def build_conda_env(
    python_version: str,
    requirements_path: Path,
) -> tuple[Path, Path, dict]:
    """Builds a conda environment given the requirements and packages it up with conda-pack and writes
    the resulting archive to the output_dir

    Args:
        python_version: the python version to use for the environment
        requirements_path: The path to the requirements file to use

    Returns:
        path to the output archive
        path to json file with info for the environment
        dictionary of the conda environment info
    """
    # Hash inputs
    req_bytes = requirements_path.read_bytes()
    pf = get_platform_desc().encode()
    hsh = calc_hash(req_bytes, pf, python_version)

    env_data = {
        "python_version": python_version,
        "requirements": req_bytes.decode().split("\n"),
        "platform": pf.decode(),
        "hash": hsh,
        "build_version": CONDA_BUILD_VERSION,
    }

    output_dir = TSBK_DIR.joinpath("conda_packs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir.joinpath(f"{hsh}.tar.gz")
    info_path = output_dir.joinpath(f"{hsh}.json")

    if output_path.exists() and info_path.exists():
        return output_path, info_path, env_data

    click.secho(f"Running conda build script for {requirements_path} on platform {pf.decode()}", fg="blue")
    _build_conda_pack(
        python_version=python_version,
        requirements=requirements_path,
        output_path=output_path,
    )

    with open(info_path, "w") as fp:
        json.dump(env_data, fp)

    return output_path, info_path, env_data
