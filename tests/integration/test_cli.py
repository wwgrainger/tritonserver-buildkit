import pytest
from click import Context

from tsbk.cli import build, create_test_plan, run, run_test_plan, test


def test_build(assets_dir, model_repo_dir, s3_pytorch_model):
    Context(build).invoke(
        build, config_file=assets_dir.joinpath("example-repo.yaml"), output_path=model_repo_dir.joinpath("example-repo")
    )
    assert model_repo_dir.joinpath("example-repo").exists()
    assert model_repo_dir.joinpath("example-repo/s3-model/config.pbtxt").exists()
    assert model_repo_dir.joinpath("example-repo/s3-model/1/model.pt").exists()


def test_run(assets_dir, model_repo_dir, s3_pytorch_model):
    with pytest.raises(SystemExit) as e:
        Context(run).invoke(
            run,
            config_file=assets_dir.joinpath("example-repo.yaml"),
            repo_path=model_repo_dir.joinpath("example-repo"),
            http_port=8000,
            grpc_port=8001,
            metrics_port=8002,
            test=True,
        )
    assert e.value.code == 0


def test_test(assets_dir, model_repo_dir, s3_pytorch_model):
    Context(run).invoke(
        run,
        config_file=assets_dir.joinpath("example-repo.yaml"),
        repo_path=model_repo_dir.joinpath("example-repo"),
        http_port=10000,
        grpc_port=10001,
        metrics_port=10002,
        detach=True,
    )
    with pytest.raises(SystemExit) as e:
        Context(test).invoke(
            test,
            config_file=assets_dir.joinpath("example-repo.yaml"),
            repo_path=model_repo_dir.joinpath("example-repo"),
            url="http://localhost:10000",
            grpc=False,
        )
    assert e.value.code == 0

    with pytest.raises(SystemExit) as e:
        Context(test).invoke(
            test,
            config_file=assets_dir.joinpath("example-repo.yaml"),
            repo_path=model_repo_dir.joinpath("example-repo"),
            url="http://localhost:10001",
            grpc=True,
        )
    assert e.value.code == 0


def test_test_with_plan(assets_dir, model_repo_dir, tmp_path, s3_pytorch_model):
    Context(create_test_plan).invoke(
        create_test_plan,
        config_file=assets_dir.joinpath("example-repo.yaml"),
        repo_path=model_repo_dir.joinpath("example-repo"),
        output_path=tmp_path.joinpath("test_plan.msgpack"),
    )
    assert tmp_path.joinpath("test_plan.msgpack").exists()

    Context(run).invoke(
        run,
        config_file=assets_dir.joinpath("example-repo.yaml"),
        repo_path=model_repo_dir.joinpath("example-repo"),
        http_port=10003,
        grpc_port=10004,
        metrics_port=10005,
        detach=True,
    )

    with pytest.raises(SystemExit) as e:
        Context(run_test_plan).invoke(
            run_test_plan,
            test_plan_path=tmp_path.joinpath("test_plan.msgpack"),
            url="http://localhost:10003",
            grpc=False,
        )
    assert e.value.code == 0

    with pytest.raises(SystemExit) as e:
        Context(run_test_plan).invoke(
            run_test_plan,
            test_plan_path=tmp_path.joinpath("test_plan.msgpack"),
            url="http://localhost:10004",
            grpc=True,
        )
    assert e.value.code == 0
