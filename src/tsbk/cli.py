import json
import sys

import click
import msgpack
import yaml

from tsbk import __version__
from tsbk.model_repo import TritonModelRepo
from tsbk.spec import TritonModelRepoSpec


@click.group()
def tsbk():
    pass


@tsbk.command(help="Print current tsbk version")
def version():
    click.echo(f"tsbk version: {__version__}")


@tsbk.command(help="Build a Tritonserver model repository from a config file")
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path(exists=False))
def build(config_file, output_path):
    """Build a Tritonserver model repository from a config file."""
    with open(config_file, "r") as f:
        click.secho(f"Reading configuration from: {config_file}")
        config = yaml.safe_load(f)

    spec = TritonModelRepoSpec(**config)
    repo = TritonModelRepo(path=output_path, **spec.model_dump())
    repo.build()
    click.secho(f"Model repository built at: {repo.path}")


@tsbk.command(help="Run a Tritonserver model repository from a config file")
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("repo_path", type=click.Path(exists=False))
@click.option("--http-port", default=8000, show_default=True, help="HTTP port for Triton server")
@click.option("--grpc-port", default=8001, show_default=True, help="gRPC port for Triton server")
@click.option("--metrics-port", default=8002, show_default=True, help="Metrics port for Triton server")
@click.option("--shm-size", default="4G", show_default=True, help="Shared memory size for Docker container")
@click.option("--gpus", is_flag=True, help="Enable GPU support")
@click.option("--env-file", type=click.Path(exists=True, dir_okay=False), default=None, help="Path to Docker env file")
@click.option("--detach", is_flag=True, help="Run container in detached mode")
@click.option("--test", is_flag=True, help="Run container in detached mode")
def run(config_file, repo_path, http_port, grpc_port, metrics_port, shm_size, gpus, env_file, detach, test):
    """Run a Tritonserver model repository from a config file."""
    if test:
        detach = True
    with open(config_file, "r") as f:
        click.secho(f"Reading configuration from: {config_file}")
        config = yaml.safe_load(f)
    spec = TritonModelRepoSpec(**config)
    repo = TritonModelRepo(path=repo_path, **spec.model_dump())
    repo.build()
    repo.run(
        http_port=http_port,
        grpc_port=grpc_port,
        metrics_port=metrics_port,
        shm_size=shm_size,
        gpus=gpus,
        env_file=env_file,
        detach=detach,
    )
    if detach:
        click.secho(f"Triton server started for repo at: {repo.path}")

    if test:
        click.secho("Running tests for the Triton server repository...")
        http_results = repo.test(url=repo.http_url)
        grpc_results = repo.test(url=repo.grpc_url, grpc=True)
        all_results = http_results + grpc_results
        success = all(r.success for r in all_results)
        repo.stop()
        sys.exit(0 if success else 1)


@tsbk.command(help="Test a Tritonserver model repository against a running server")
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("repo_path", type=click.Path(exists=False))
@click.option("--url", required=True, help="URL of the running Triton server (http or grpc)")
@click.option(
    "--ca-certs",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to CA certificates for secure connections",
)
@click.option("--headers", type=str, default=None, help="Additional headers as JSON string")
@click.option("--grpc", is_flag=True, help="Use gRPC for testing")
def test(config_file, repo_path, url, ca_certs, headers, grpc):
    """Test a Tritonserver model repository against a running server."""
    with open(config_file, "r") as f:
        click.secho(f"Reading configuration from: {config_file}")
        config = yaml.safe_load(f)
    spec = TritonModelRepoSpec(**config)
    repo = TritonModelRepo(path=repo_path, **spec.model_dump())
    headers_dict = json.loads(headers) if headers else None
    click.secho(f"Testing Triton server at: {url}", fg="blue")
    results = repo.test(url=url, ca_certs=ca_certs, headers=headers_dict, grpc=grpc)
    success = all(r.success for r in results)
    for r in results:
        click.secho(
            f"Test for model {r.model_name} (version {r.model_version}): {'PASS' if r.success else 'FAIL'}",
            fg="green" if r.success else "red",
        )
    sys.exit(0 if success else 1)


@tsbk.command(help="Create a test plan for a Tritonserver model repository and serialize it to a file")
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("repo_path", type=click.Path(exists=False))
@click.argument("output_path", type=click.Path(exists=False))
def create_test_plan(config_file, repo_path, output_path):
    """Create a test plan and serialize it to a file."""
    with open(config_file, "r") as f:
        click.secho(f"Reading configuration from: {config_file}")
        config = yaml.safe_load(f)
    spec = TritonModelRepoSpec(**config)
    repo = TritonModelRepo(path=repo_path, **spec.model_dump())
    test_plan = repo.create_test_plan()
    with open(output_path, "wb") as f:
        f.write(msgpack.packb(test_plan.to_dict()))
    click.secho(f"Test plan created and saved to: {output_path}", fg="green")


@tsbk.command(help="Run a serialized test plan against a running Tritonserver")
@click.argument("test_plan_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--url", required=True, help="URL of the running Triton server (http or grpc)")
@click.option(
    "--ca-certs",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to CA certificates for secure connections",
)
@click.option("--headers", type=str, default=None, help="Additional headers as JSON string")
@click.option("--grpc", is_flag=True, help="Use gRPC for testing")
def run_test_plan(test_plan_path, url, ca_certs, headers, grpc):
    """Run a serialized test plan against a running Tritonserver."""
    import json

    from tsbk.testing import TritonModelRepoTestPlan

    with open(test_plan_path, "rb") as f:
        test_plan_dict = msgpack.unpackb(f.read(), raw=False)
        test_plan = TritonModelRepoTestPlan.from_dict(test_plan_dict)
    headers_dict = json.loads(headers) if headers else None
    click.secho(f"Running test plan against Triton server at: {url}", fg="blue")
    results = test_plan.run_tests(url=url, ca_certs=ca_certs, headers=headers_dict, grpc=grpc)
    success = all(r.success for r in results)
    for r in results:
        click.secho(
            f"Test for model {r.model_name} (version {r.model_version}): {'PASS' if r.success else 'FAIL'}",
            fg="green" if r.success else "red",
        )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    tsbk()
