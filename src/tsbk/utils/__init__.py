import os
import platform
import shutil
from pathlib import Path

import click


def get_platform_desc() -> str:
    """Returns the description for the current platform"""
    return f"{platform.system()}-{platform.machine()}"


def parse_requirements_file(requirements_path: Path) -> set[str]:
    """Parses all the requirements listed in the requirements file"""
    with open(requirements_path, "r") as fp:
        reqs = fp.readlines()
        reqs = [r.strip() for r in reqs]
        return set(reqs) - {""}


def parse_python_version(python_version: str) -> str:
    """Parses the passed python version ensures it takes the form of major.minor version (i.e. removing any patch version provided)"""
    try:
        major, minor = python_version.split(".")
        return f"{major}.{minor}"
    except ValueError:
        major, minor, _ = python_version.split(".")
        return f"{major}.{minor}"


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except (PermissionError, OSError) as e:
        # If hard linking fails due to permission issues, fall back to copying
        click.secho(f"Hard linking failed with exception {e}, falling back to copying.", fg="yellow")
        shutil.copy2(src, dst)
