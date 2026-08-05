import hashlib
import platform as _platform
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from tsbk.utils.trtexec import (
    DEFAULT_TRT_IMAGE,
    _build_trtexec_command,
    _can_compile_docker,
    _can_compile_kubernetes,
    _compile_docker,
    _create_trt_job_manifest,
    _find_onnx_file,
    build_trt_engine,
)

# ---------------------------------------------------------------------------
# _build_trtexec_command
# ---------------------------------------------------------------------------


class TestBuildTrtexecCommand:
    def test_basic_command(self):
        cmd = _build_trtexec_command("/workspace/model.onnx", "/workspace/model.plan")
        assert cmd == "trtexec --onnx=/workspace/model.onnx --saveEngine=/workspace/model.plan"

    def test_fp16_precision(self):
        cmd = _build_trtexec_command("/workspace/model.onnx", "/workspace/model.plan", precision="fp16")
        assert "--fp16" in cmd
        assert "--int8" not in cmd
        assert "--best" not in cmd

    def test_int8_precision(self):
        cmd = _build_trtexec_command("/workspace/model.onnx", "/workspace/model.plan", precision="int8")
        assert "--int8" in cmd
        assert "--fp16" not in cmd

    def test_best_precision(self):
        cmd = _build_trtexec_command("/workspace/model.onnx", "/workspace/model.plan", precision="best")
        assert "--best" in cmd

    def test_workspace_size(self):
        cmd = _build_trtexec_command("/workspace/model.onnx", "/workspace/model.plan", workspace_size=4096)
        assert "--workspace=4096" in cmd

    def test_workspace_size_zero(self):
        cmd = _build_trtexec_command("/workspace/model.onnx", "/workspace/model.plan", workspace_size=0)
        assert "--workspace=0" in cmd

    def test_extra_args(self):
        cmd = _build_trtexec_command(
            "/workspace/model.onnx", "/workspace/model.plan", extra_args="--verbose --minShapes=input:1x3x224x224"
        )
        assert "--verbose --minShapes=input:1x3x224x224" in cmd

    def test_all_options_combined(self):
        cmd = _build_trtexec_command(
            "/workspace/model.onnx",
            "/workspace/model.plan",
            precision="fp16",
            workspace_size=2048,
            extra_args="--verbose",
        )
        assert cmd.startswith("trtexec --onnx=/workspace/model.onnx --saveEngine=/workspace/model.plan")
        assert "--fp16" in cmd
        assert "--workspace=2048" in cmd
        assert "--verbose" in cmd

    def test_invalid_precision_raises(self):
        with pytest.raises(ValueError, match="Unknown precision mode"):
            _build_trtexec_command("/workspace/model.onnx", "/workspace/model.plan", precision="fp64")

    def test_empty_string_precision_raises(self):
        with pytest.raises(ValueError, match="Unknown precision mode"):
            _build_trtexec_command("/workspace/model.onnx", "/workspace/model.plan", precision="")

    def test_none_precision_no_flag(self):
        cmd = _build_trtexec_command("/workspace/model.onnx", "/workspace/model.plan", precision=None)
        assert "--fp16" not in cmd
        assert "--int8" not in cmd
        assert "--best" not in cmd

    def test_empty_extra_args_ignored(self):
        cmd = _build_trtexec_command("/workspace/model.onnx", "/workspace/model.plan", extra_args="")
        assert cmd == "trtexec --onnx=/workspace/model.onnx --saveEngine=/workspace/model.plan"

    def test_none_workspace_no_flag(self):
        cmd = _build_trtexec_command("/workspace/model.onnx", "/workspace/model.plan", workspace_size=None)
        assert "--workspace" not in cmd

    def test_flag_ordering(self):
        """Precision flag comes before workspace which comes before extra_args."""
        cmd = _build_trtexec_command(
            "/workspace/model.onnx",
            "/workspace/model.plan",
            precision="fp16",
            workspace_size=1024,
            extra_args="--verbose",
        )
        fp16_idx = cmd.index("--fp16")
        ws_idx = cmd.index("--workspace")
        verbose_idx = cmd.index("--verbose")
        assert fp16_idx < ws_idx < verbose_idx


# ---------------------------------------------------------------------------
# _find_onnx_file
# ---------------------------------------------------------------------------


class TestFindOnnxFile:
    def test_finds_flat_onnx(self, tmp_path):
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")
        assert _find_onnx_file(tmp_path) == onnx_file

    def test_finds_nested_onnx(self, tmp_path):
        subdir = tmp_path / "data"
        subdir.mkdir()
        onnx_file = subdir / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")
        assert _find_onnx_file(tmp_path) == onnx_file

    def test_finds_deeply_nested_onnx(self, tmp_path):
        subdir = tmp_path / "artifacts" / "data" / "model"
        subdir.mkdir(parents=True)
        onnx_file = subdir / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")
        assert _find_onnx_file(tmp_path) == onnx_file

    def test_raises_when_no_onnx(self, tmp_path):
        (tmp_path / "model.plan").write_bytes(b"fake plan")
        with pytest.raises(RuntimeError, match="Cannot find ONNX model file"):
            _find_onnx_file(tmp_path)

    def test_raises_on_empty_directory(self, tmp_path):
        with pytest.raises(RuntimeError, match="Cannot find ONNX model file"):
            _find_onnx_file(tmp_path)

    def test_returns_one_when_multiple_flat(self, tmp_path):
        """When multiple .onnx files exist in the flat directory, returns the first one."""
        (tmp_path / "a_model.onnx").write_bytes(b"onnx1")
        (tmp_path / "b_model.onnx").write_bytes(b"onnx2")
        result = _find_onnx_file(tmp_path)
        assert result.suffix == ".onnx"

    def test_prefers_flat_over_nested(self, tmp_path):
        """Flat .onnx files should be found before nested ones."""
        flat_file = tmp_path / "model.onnx"
        flat_file.write_bytes(b"flat onnx")

        subdir = tmp_path / "data"
        subdir.mkdir()
        (subdir / "nested.onnx").write_bytes(b"nested onnx")

        result = _find_onnx_file(tmp_path)
        assert result == flat_file

    def test_ignores_non_onnx_files(self, tmp_path):
        (tmp_path / "model.pt").write_bytes(b"pytorch")
        (tmp_path / "model.plan").write_bytes(b"tensorrt")
        (tmp_path / "model.onnx.backup").write_bytes(b"backup")
        with pytest.raises(RuntimeError, match="Cannot find ONNX model file"):
            _find_onnx_file(tmp_path)


# ---------------------------------------------------------------------------
# _can_compile_docker
# ---------------------------------------------------------------------------


class TestCanCompileDocker:
    # Helper: default which side_effect that finds docker and nvidia-smi
    @staticmethod
    def _which_side_effect(cmd):
        return {
            "docker": "/usr/bin/docker",
            "nvidia-smi": "/usr/bin/nvidia-smi",
        }.get(cmd)

    @staticmethod
    def _docker_info_nvidia():
        """A MagicMock resembling docker info output with nvidia runtime."""
        return MagicMock(returncode=0, stdout="Runtimes: io.containerd.runc.v2 nvidia runc\n", stderr="")

    @staticmethod
    def _nvidia_smi_result(gpu_names: list[str] | None = None):
        """A MagicMock resembling nvidia-smi --query-gpu=gpu_name output."""
        names = gpu_names or ["NVIDIA A10G"]
        return MagicMock(returncode=0, stdout="\n".join(names) + "\n", stderr="")

    @patch("tsbk.utils.trtexec.shutil.which", return_value=None)
    def test_no_docker_binary(self, mock_which):
        ok, reason = _can_compile_docker()
        assert ok is False
        assert "docker command not found" in reason

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which", return_value="/usr/bin/docker")
    def test_docker_info_fails(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="Cannot connect to daemon")
        ok, reason = _can_compile_docker()
        assert ok is False
        assert "docker info failed" in reason

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which", return_value="/usr/bin/docker")
    def test_docker_info_timeout(self, mock_which, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker info", timeout=10)
        ok, reason = _can_compile_docker()
        assert ok is False
        assert "timed out" in reason

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which")
    def test_no_nvidia_runtime_in_docker(self, mock_which, mock_run):
        """Docker is running but has no nvidia runtime configured."""
        mock_which.side_effect = self._which_side_effect
        mock_run.return_value = MagicMock(returncode=0, stdout="Runtimes: runc\n", stderr="")
        ok, reason = _can_compile_docker()
        assert ok is False
        assert "NVIDIA runtime" in reason

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which")
    def test_no_nvidia_smi_binary(self, mock_which, mock_run):
        """Docker has nvidia runtime but nvidia-smi is not installed."""
        mock_which.side_effect = lambda cmd: "/usr/bin/docker" if cmd == "docker" else None
        mock_run.return_value = self._docker_info_nvidia()
        ok, reason = _can_compile_docker()
        assert ok is False
        assert "nvidia-smi not found" in reason

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which")
    def test_nvidia_smi_fails(self, mock_which, mock_run):
        mock_which.side_effect = self._which_side_effect
        mock_run.side_effect = [
            self._docker_info_nvidia(),
            MagicMock(returncode=1, stdout="", stderr="NVIDIA-SMI has failed"),
        ]
        ok, reason = _can_compile_docker()
        assert ok is False
        assert "nvidia-smi failed" in reason

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which")
    def test_nvidia_smi_timeout(self, mock_which, mock_run):
        mock_which.side_effect = self._which_side_effect
        mock_run.side_effect = [
            self._docker_info_nvidia(),
            subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10),
        ]
        ok, reason = _can_compile_docker()
        assert ok is False
        assert "nvidia-smi timed out" in reason

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which")
    def test_nvidia_smi_no_gpus(self, mock_which, mock_run):
        mock_which.side_effect = self._which_side_effect
        mock_run.side_effect = [
            self._docker_info_nvidia(),
            MagicMock(returncode=0, stdout="\n", stderr=""),
        ]
        ok, reason = _can_compile_docker()
        assert ok is False
        assert "no GPUs" in reason

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which")
    def test_gpu_name_mismatch(self, mock_which, mock_run):
        mock_which.side_effect = self._which_side_effect
        mock_run.side_effect = [
            self._docker_info_nvidia(),
            self._nvidia_smi_result(["NVIDIA T4"]),
        ]
        ok, reason = _can_compile_docker(gpu_name="A10G")
        assert ok is False
        assert "not found" in reason
        assert "NVIDIA T4" in reason

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which")
    def test_gpu_name_match(self, mock_which, mock_run):
        mock_which.side_effect = self._which_side_effect
        mock_run.side_effect = [
            self._docker_info_nvidia(),
            self._nvidia_smi_result(["NVIDIA A10G"]),
        ]
        ok, reason = _can_compile_docker(gpu_name="A10G")
        assert ok is True
        assert "NVIDIA A10G" in reason

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which")
    def test_gpu_name_match_case_insensitive(self, mock_which, mock_run):
        mock_which.side_effect = self._which_side_effect
        mock_run.side_effect = [
            self._docker_info_nvidia(),
            self._nvidia_smi_result(["NVIDIA A10G"]),
        ]
        ok, reason = _can_compile_docker(gpu_name="a10g")
        assert ok is True

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which")
    def test_gpu_name_match_multi_gpu(self, mock_which, mock_run):
        """Passes if any detected GPU matches the requested name."""
        mock_which.side_effect = self._which_side_effect
        mock_run.side_effect = [
            self._docker_info_nvidia(),
            self._nvidia_smi_result(["NVIDIA T4", "NVIDIA A10G"]),
        ]
        ok, reason = _can_compile_docker(gpu_name="A10G")
        assert ok is True

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which")
    def test_docker_available(self, mock_which, mock_run):
        mock_which.side_effect = self._which_side_effect
        mock_run.side_effect = [
            self._docker_info_nvidia(),
            self._nvidia_smi_result(["NVIDIA A10G"]),
        ]
        ok, reason = _can_compile_docker()
        assert ok is True
        assert "available" in reason.lower()

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which")
    def test_s3_path_not_required(self, mock_which, mock_run):
        """Docker method does not require an S3 path."""
        mock_which.side_effect = self._which_side_effect
        mock_run.side_effect = [
            self._docker_info_nvidia(),
            self._nvidia_smi_result(),
        ]
        ok, _ = _can_compile_docker(shared_s3_path=None)
        assert ok is True


# ---------------------------------------------------------------------------
# _can_compile_kubernetes
# ---------------------------------------------------------------------------


class TestCanCompileKubernetes:
    @patch("tsbk.utils.trtexec.shutil.which", return_value=None)
    def test_no_kubectl_binary(self, mock_which):
        ok, reason = _can_compile_kubernetes(shared_s3_path="s3://bucket/key")
        assert ok is False
        assert "kubectl command not found" in reason

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which", return_value="/usr/bin/kubectl")
    def test_cluster_unreachable(self, mock_which, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "kubectl get pods")
        ok, reason = _can_compile_kubernetes(shared_s3_path="s3://bucket/key")
        assert ok is False
        assert "Unable to connect" in reason

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which", return_value="/usr/bin/kubectl")
    def test_cluster_timeout(self, mock_which, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="kubectl get pods", timeout=10)
        ok, reason = _can_compile_kubernetes(shared_s3_path="s3://bucket/key")
        assert ok is False
        assert "Unable to connect" in reason

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which", return_value="/usr/bin/kubectl")
    def test_no_s3_path(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="pod1\npod2")
        ok, reason = _can_compile_kubernetes(shared_s3_path=None)
        assert ok is False
        assert "TSBK_S3_PREFIX" in reason

    @patch("tsbk.utils.trtexec.subprocess.run")
    @patch("tsbk.utils.trtexec.shutil.which", return_value="/usr/bin/kubectl")
    def test_all_available(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="pod1\npod2")
        ok, reason = _can_compile_kubernetes(shared_s3_path="s3://bucket/key")
        assert ok is True
        assert "available" in reason.lower()


# ---------------------------------------------------------------------------
# _compile_docker
# ---------------------------------------------------------------------------


class TestCompileDocker:
    @patch("tsbk.utils.trtexec.shutil.move")
    @patch("tsbk.utils.trtexec.subprocess.run")
    def test_basic_docker_run(self, mock_run, mock_move, tmp_path):
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")
        output = tmp_path / "cache" / "out.plan"
        output.parent.mkdir()

        mock_run.return_value = MagicMock(returncode=0)

        _compile_docker(
            onnx_path=onnx_file,
            output_path=output,
            trt_image="nvcr.io/nvidia/tensorrt:test-py3",
        )

        # Check docker run was called
        args = mock_run.call_args
        cmd = args.kwargs.get("args") or args[1].get("args") if args[1] else args[0][0]
        if not isinstance(cmd, list):
            cmd = args[0][0] if args[0] else args.kwargs["args"]
        assert "docker" in cmd
        assert "--gpus" in cmd
        assert "all" in cmd
        assert "nvcr.io/nvidia/tensorrt:test-py3" in cmd

    @patch("tsbk.utils.trtexec.subprocess.run")
    def test_docker_failure_raises(self, mock_run, tmp_path):
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")
        output = tmp_path / "out.plan"

        mock_run.return_value = MagicMock(returncode=1)

        with pytest.raises(RuntimeError, match="TensorRT engine compilation failed in Docker"):
            _compile_docker(onnx_path=onnx_file, output_path=output)

    @patch("tsbk.utils.trtexec.shutil.move")
    @patch("tsbk.utils.trtexec.subprocess.run")
    def test_docker_passes_precision(self, mock_run, mock_move, tmp_path):
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        mock_run.return_value = MagicMock(returncode=0)

        _compile_docker(
            onnx_path=onnx_file,
            output_path=tmp_path / "out.plan",
            precision="fp16",
            workspace_size=2048,
        )

        cmd = mock_run.call_args.kwargs.get("args") or mock_run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        assert "--fp16" in cmd_str
        assert "--workspace=2048" in cmd_str

    @patch("tsbk.utils.trtexec.shutil.move")
    @patch("tsbk.utils.trtexec.subprocess.run")
    def test_docker_mounts_onnx_parent(self, mock_run, mock_move, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        onnx_file = model_dir / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        mock_run.return_value = MagicMock(returncode=0)

        _compile_docker(onnx_path=onnx_file, output_path=tmp_path / "out.plan")

        cmd = mock_run.call_args.kwargs.get("args") or mock_run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        assert f"{model_dir}:/workspace" in cmd_str

    @patch("tsbk.utils.trtexec.shutil.move")
    @patch("tsbk.utils.trtexec.subprocess.run")
    def test_docker_ca_bundle_mount(self, mock_run, mock_move, tmp_path, monkeypatch):
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")
        mock_run.return_value = MagicMock(returncode=0)

        _compile_docker(onnx_path=onnx_file, output_path=tmp_path / "out.plan")

        cmd = mock_run.call_args.kwargs.get("args") or mock_run.call_args[0][0]
        cmd_str = " ".join(str(c) for c in cmd)
        assert "/etc/ssl/certs/ca-certificates.crt:/tmp/requests_ca_bundle.pem" in cmd_str
        assert "REQUESTS_CA_BUNDLE=/tmp/requests_ca_bundle.pem" in cmd_str

    @patch("tsbk.utils.trtexec.shutil.move")
    @patch("tsbk.utils.trtexec.subprocess.run")
    def test_docker_moves_plan_to_output(self, mock_run, mock_move, tmp_path):
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")
        output = tmp_path / "cache" / "abc123.plan"
        output.parent.mkdir()

        mock_run.return_value = MagicMock(returncode=0)

        _compile_docker(onnx_path=onnx_file, output_path=output)

        # shutil.move should move model.plan from onnx parent to output
        mock_move.assert_called_once_with(
            str(onnx_file.parent / "model.plan"),
            str(output),
        )


# ---------------------------------------------------------------------------
# _create_trt_job_manifest
# ---------------------------------------------------------------------------


class TestCreateTrtJobManifest:
    def test_basic_manifest_structure(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://bucket/models/model_model/",
            s3_plan_path="s3://bucket/engines/model.plan",
        )
        assert manifest["apiVersion"] == "batch/v1"
        assert manifest["kind"] == "Job"
        assert manifest["metadata"]["name"] == "test-job"
        assert manifest["spec"]["backoffLimit"] == 0
        assert manifest["spec"]["ttlSecondsAfterFinished"] == 3600

    def test_karpenter_annotation(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://b/m_model/",
            s3_plan_path="s3://b/m.plan",
        )
        assert manifest["metadata"]["annotations"]["karpenter.sh/do-not-disrupt"] == "true"

    def test_gpu_toleration(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://b/m_model/",
            s3_plan_path="s3://b/m.plan",
        )
        tolerations = manifest["spec"]["template"]["spec"]["tolerations"]
        assert len(tolerations) == 1
        assert tolerations[0] == {"effect": "NoSchedule", "key": "gpu", "value": "true"}

    def test_gpu_resource_requests(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://b/m_model/",
            s3_plan_path="s3://b/m.plan",
        )
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"]["requests"]["nvidia.com/gpu"] == "1"
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"

    def test_default_resource_values(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://b/m_model/",
            s3_plan_path="s3://b/m.plan",
        )
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"]["requests"]["cpu"] == "2"
        assert container["resources"]["requests"]["memory"] == "8Gi"
        assert container["resources"]["limits"]["memory"] == "16Gi"

    def test_custom_resources(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://b/m_model/",
            s3_plan_path="s3://b/m.plan",
            cpu="4",
            memory="16Gi",
            memory_limit="32Gi",
        )
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"]["requests"]["cpu"] == "4"
        assert container["resources"]["requests"]["memory"] == "16Gi"
        assert container["resources"]["limits"]["memory"] == "32Gi"

    def test_gpu_name_sets_node_selector(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://b/m_model/",
            s3_plan_path="s3://b/m.plan",
            gpu_name="A10G",
        )
        node_selector = manifest["spec"]["template"]["spec"]["nodeSelector"]
        assert node_selector == {"karpenter.k8s.aws/instance-gpu-name": "a10g"}

    def test_gpu_name_lowercased(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://b/m_model/",
            s3_plan_path="s3://b/m.plan",
            gpu_name="T4",
        )
        node_selector = manifest["spec"]["template"]["spec"]["nodeSelector"]
        assert node_selector["karpenter.k8s.aws/instance-gpu-name"] == "t4"

    def test_no_gpu_name_no_node_selector(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://b/m_model/",
            s3_plan_path="s3://b/m.plan",
            gpu_name=None,
        )
        assert "nodeSelector" not in manifest["spec"]["template"]["spec"]

    def test_service_account(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://b/m_model/",
            s3_plan_path="s3://b/m.plan",
            service_account="my-sa",
        )
        assert manifest["spec"]["template"]["spec"]["serviceAccountName"] == "my-sa"

    def test_container_image(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://b/m_model/",
            s3_plan_path="s3://b/m.plan",
            trt_image="custom-image:latest",
        )
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == "custom-image:latest"

    def test_compile_script_contains_s3_paths(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://my-bucket/tmp/model_model/",
            s3_plan_path="s3://my-bucket/engines/model.plan",
        )
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        script = container["command"][2]
        assert "aws s3 cp --recursive s3://my-bucket/tmp/model_model/ /workspace/" in script
        assert "s3://my-bucket/engines/model.plan" in script

    def test_compile_script_uses_onnx_filename(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://b/m_model/",
            s3_plan_path="s3://b/m.plan",
            onnx_filename="resnet50.onnx",
        )
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        script = container["command"][2]
        assert "--onnx=/workspace/resnet50.onnx" in script

    def test_compile_script_default_onnx_filename(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://b/m_model/",
            s3_plan_path="s3://b/m.plan",
        )
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        script = container["command"][2]
        assert "--onnx=/workspace/model.onnx" in script

    def test_compile_script_contains_trtexec(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://b/m_model/",
            s3_plan_path="s3://b/m.plan",
            precision="fp16",
            workspace_size=4096,
        )
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        script = container["command"][2]
        assert "trtexec" in script
        assert "--fp16" in script
        assert "--workspace=4096" in script

    def test_compile_script_has_set_ex(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://b/m_model/",
            s3_plan_path="s3://b/m.plan",
        )
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        script = container["command"][2]
        assert script.startswith("set -ex")

    def test_job_labels(self):
        manifest = _create_trt_job_manifest(
            job_name="my-compile-job",
            s3_model_dir="s3://b/m_model/",
            s3_plan_path="s3://b/m.plan",
        )
        labels = manifest["spec"]["template"]["metadata"]["labels"]
        assert labels["tsbk-job-id"] == "my-compile-job"

    def test_restart_policy_never(self):
        manifest = _create_trt_job_manifest(
            job_name="test-job",
            s3_model_dir="s3://b/m_model/",
            s3_plan_path="s3://b/m.plan",
        )
        assert manifest["spec"]["template"]["spec"]["restartPolicy"] == "Never"


# ---------------------------------------------------------------------------
# build_trt_engine — caching
# ---------------------------------------------------------------------------


class TestBuildTrtEngineCaching:
    def _compute_cache_key(
        self,
        onnx_content,
        precision=None,
        workspace_size=None,
        extra_args=None,
        gpu_name=None,
        instance_family=None,
        cache_bust=None,
    ):
        arch = _platform.machine()
        onnx_hash = hashlib.sha256(onnx_content).hexdigest()[:16]
        params_str = f"{precision or 'default'}-{workspace_size or 'default'}-{extra_args or ''}-{gpu_name or 'any'}-{instance_family or 'any'}-{arch}"
        cache_bust_material = b"" if not cache_bust else b"\x00cache_bust\x00" + cache_bust.encode()
        params_hash = hashlib.sha256(params_str.encode() + cache_bust_material).hexdigest()[:8]
        return f"{onnx_hash}-{params_hash}"

    def test_cache_hit_returns_immediately(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx content")

        cache_key = self._compute_cache_key(b"fake onnx content")
        cache_dir = tmp_path / "trt_engines"
        cache_dir.mkdir()
        cached_plan = cache_dir / f"{cache_key}.plan"
        cached_plan.write_bytes(b"cached plan")

        result = build_trt_engine(onnx_file)
        assert result == cached_plan
        assert result.read_bytes() == b"cached plan"

    def test_cache_hit_with_precision(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"onnx data")

        cache_key = self._compute_cache_key(b"onnx data", precision="fp16")
        cache_dir = tmp_path / "trt_engines"
        cache_dir.mkdir()
        (cache_dir / f"{cache_key}.plan").write_bytes(b"fp16 plan")

        result = build_trt_engine(onnx_file, precision="fp16")
        assert result.read_bytes() == b"fp16 plan"

    def test_different_params_different_cache_keys(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        content = b"fake onnx content"
        key_fp16 = self._compute_cache_key(content, precision="fp16")
        key_default = self._compute_cache_key(content)
        key_int8 = self._compute_cache_key(content, precision="int8")

        assert key_fp16 != key_default
        assert key_fp16 != key_int8
        assert key_default != key_int8

    def test_different_onnx_different_cache_keys(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        key1 = self._compute_cache_key(b"onnx content 1")
        key2 = self._compute_cache_key(b"onnx content 2")
        assert key1 != key2

    def test_gpu_name_affects_cache_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        content = b"fake onnx"
        key_a10g = self._compute_cache_key(content, gpu_name="A10G")
        key_t4 = self._compute_cache_key(content, gpu_name="T4")
        key_none = self._compute_cache_key(content)

        assert key_a10g != key_t4
        assert key_a10g != key_none

    def test_workspace_size_affects_cache_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        content = b"fake onnx"
        key_4096 = self._compute_cache_key(content, workspace_size=4096)
        key_2048 = self._compute_cache_key(content, workspace_size=2048)
        key_none = self._compute_cache_key(content)

        assert key_4096 != key_2048
        assert key_4096 != key_none

    def test_extra_args_affects_cache_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        content = b"fake onnx"
        key_verbose = self._compute_cache_key(content, extra_args="--verbose")
        key_none = self._compute_cache_key(content)

        assert key_verbose != key_none

    def test_same_params_same_cache_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        content = b"fake onnx"
        key1 = self._compute_cache_key(content, precision="fp16", workspace_size=2048, gpu_name="A10G")
        key2 = self._compute_cache_key(content, precision="fp16", workspace_size=2048, gpu_name="A10G")

        assert key1 == key2

    def test_cache_bust_selects_distinct_cached_engine(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx content")
        first_key = self._compute_cache_key(b"fake onnx content", cache_bust="release-1")
        second_key = self._compute_cache_key(b"fake onnx content", cache_bust="release-2")
        cache_dir = tmp_path / "trt_engines"
        cache_dir.mkdir()
        first_plan = cache_dir / f"{first_key}.plan"
        second_plan = cache_dir / f"{second_key}.plan"
        first_plan.write_bytes(b"first")
        second_plan.write_bytes(b"second")

        assert build_trt_engine(onnx_file, cache_bust="release-1") == first_plan
        assert build_trt_engine(onnx_file, cache_bust="release-2") == second_plan


# ---------------------------------------------------------------------------
# build_trt_engine — strategy pattern (fallback / method selection)
# ---------------------------------------------------------------------------


class TestBuildTrtEngineStrategy:
    def test_invalid_method_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"data")

        with pytest.raises(ValueError, match="Unknown method"):
            build_trt_engine(onnx_file, preferred_methods=["local"])

    @patch("tsbk.utils.trtexec._compile_docker")
    @patch("tsbk.utils.trtexec._can_compile_docker", return_value=(True, "Docker OK"))
    def test_docker_method_called_when_available(self, mock_can, mock_compile, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"data")

        result = build_trt_engine(onnx_file, preferred_methods=["docker"])
        mock_compile.assert_called_once()
        assert result.suffix == ".plan"

    @patch("tsbk.utils.trtexec._compile_kubernetes")
    @patch("tsbk.utils.trtexec._can_compile_kubernetes", return_value=(True, "K8s OK"))
    @patch("tsbk.utils.trtexec._can_compile_docker", return_value=(False, "no docker"))
    def test_falls_back_to_kubernetes(self, mock_can_docker, mock_can_k8s, mock_compile_k8s, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_S3_PREFIX", "s3://bucket/prefix")

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"data")

        result = build_trt_engine(onnx_file)
        mock_compile_k8s.assert_called_once()
        assert result.suffix == ".plan"

    @patch("tsbk.utils.trtexec._can_compile_kubernetes", return_value=(False, "no k8s"))
    @patch("tsbk.utils.trtexec._can_compile_docker", return_value=(False, "no docker"))
    def test_all_methods_fail_raises(self, mock_can_docker, mock_can_k8s, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"data")

        with pytest.raises(RuntimeError, match="Cannot compile ONNX to TensorRT"):
            build_trt_engine(onnx_file)

    @patch("tsbk.utils.trtexec._compile_docker")
    @patch("tsbk.utils.trtexec._can_compile_docker", return_value=(True, "Docker OK"))
    def test_docker_only_when_specified(self, mock_can, mock_compile, tmp_path, monkeypatch):
        """When preferred_methods=["docker"], kubernetes is never checked."""
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"data")

        build_trt_engine(onnx_file, preferred_methods=["docker"])
        mock_compile.assert_called_once()

    @patch("tsbk.utils.trtexec._compile_kubernetes")
    @patch("tsbk.utils.trtexec._can_compile_kubernetes", return_value=(True, "K8s OK"))
    def test_kubernetes_only_when_specified(self, mock_can, mock_compile, tmp_path, monkeypatch):
        """When preferred_methods=["kubernetes"], docker is never checked."""
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_S3_PREFIX", "s3://bucket/prefix")

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"data")

        build_trt_engine(onnx_file, preferred_methods=["kubernetes"])
        mock_compile.assert_called_once()

    @patch("tsbk.utils.trtexec._compile_docker")
    @patch("tsbk.utils.trtexec._can_compile_docker", return_value=(True, "Docker OK"))
    def test_compile_params_passed_through(self, mock_can, mock_compile, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"data")

        build_trt_engine(
            onnx_file,
            precision="fp16",
            workspace_size=4096,
            extra_args="--verbose",
            trt_image="custom:latest",
            gpu_name="A10G",
            preferred_methods=["docker"],
        )

        kwargs = mock_compile.call_args.kwargs
        assert kwargs["precision"] == "fp16"
        assert kwargs["workspace_size"] == 4096
        assert kwargs["extra_args"] == "--verbose"
        assert kwargs["trt_image"] == "custom:latest"
        assert kwargs["gpu_name"] == "A10G"

    @patch("tsbk.utils.trtexec._compile_docker")
    @patch("tsbk.utils.trtexec._can_compile_docker", return_value=(True, "Docker OK"))
    def test_s3_path_constructed_from_prefix(self, mock_can, mock_compile, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_S3_PREFIX", "s3://my-bucket/tsbk-cache")

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"data")

        build_trt_engine(onnx_file, preferred_methods=["docker"])

        kwargs = mock_compile.call_args.kwargs
        assert kwargs["k8s_shared_s3_path"].startswith("s3://my-bucket/tsbk-cache/trt_engines/")
        assert kwargs["k8s_shared_s3_path"].endswith(".plan")

    @patch("tsbk.utils.trtexec._compile_docker")
    @patch("tsbk.utils.trtexec._can_compile_docker", return_value=(True, "Docker OK"))
    def test_no_s3_path_when_no_prefix(self, mock_can, mock_compile, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_S3_PREFIX", None)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"data")

        build_trt_engine(onnx_file, preferred_methods=["docker"])

        kwargs = mock_compile.call_args.kwargs
        assert kwargs["k8s_shared_s3_path"] is None

    @patch("tsbk.utils.trtexec._compile_docker")
    @patch("tsbk.utils.trtexec._can_compile_docker", return_value=(True, "Docker OK"))
    def test_uses_default_trt_image(self, mock_can, mock_compile, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"data")

        build_trt_engine(onnx_file, preferred_methods=["docker"])

        kwargs = mock_compile.call_args.kwargs
        assert kwargs["trt_image"] == DEFAULT_TRT_IMAGE

    @patch("tsbk.utils.trtexec._compile_docker")
    @patch("tsbk.utils.trtexec._can_compile_docker", return_value=(True, "Docker OK"))
    def test_custom_trt_image_override(self, mock_can, mock_compile, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"data")

        build_trt_engine(onnx_file, trt_image="my-registry/trt:custom", preferred_methods=["docker"])

        kwargs = mock_compile.call_args.kwargs
        assert kwargs["trt_image"] == "my-registry/trt:custom"

    @patch("tsbk.utils.trtexec._compile_docker")
    @patch("tsbk.utils.trtexec._can_compile_docker", return_value=(True, "Docker OK"))
    def test_creates_cache_dir(self, mock_can, mock_compile, tmp_path, monkeypatch):
        monkeypatch.setattr("tsbk.utils.trtexec.TSBK_DIR", tmp_path)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"data")

        build_trt_engine(onnx_file, preferred_methods=["docker"])

        assert (tmp_path / "trt_engines").is_dir()


# ---------------------------------------------------------------------------
# TrtCompileSpec validation
# ---------------------------------------------------------------------------


class TestTrtCompileSpec:
    def test_spec_parses_all_fields(self):
        from tsbk.spec import TrtCompileSpec

        spec = TrtCompileSpec(
            enabled=True,
            trt_image="custom:latest",
            precision="fp16",
            workspace_size=4096,
            extra_args="--verbose",
            gpu_name="A10G",
        )
        assert spec.enabled is True
        assert spec.trt_image == "custom:latest"
        assert spec.precision == "fp16"
        assert spec.workspace_size == 4096
        assert spec.extra_args == "--verbose"
        assert spec.gpu_name == "A10G"

    def test_spec_defaults(self):
        from tsbk.spec import TrtCompileSpec

        spec = TrtCompileSpec()
        assert spec.enabled is False
        assert spec.trt_image is None
        assert spec.precision is None
        assert spec.workspace_size is None
        assert spec.extra_args is None
        assert spec.gpu_name is None

    def test_spec_forbids_extra_fields(self):
        from pydantic import ValidationError

        from tsbk.spec import TrtCompileSpec

        with pytest.raises(ValidationError):
            TrtCompileSpec(enabled=True, unknown_field="value")

    def test_version_spec_with_trt_compile(self):
        from tsbk.spec import TritonModelVersionSpec

        spec = TritonModelVersionSpec(
            artifact_uri="s3://bucket/model.onnx",
            trt_compile={"enabled": True, "precision": "fp16", "gpu_name": "T4"},
        )
        assert spec.trt_compile.enabled is True
        assert spec.trt_compile.precision == "fp16"
        assert spec.trt_compile.gpu_name == "T4"

    def test_version_spec_without_trt_compile(self):
        from tsbk.spec import TritonModelVersionSpec

        spec = TritonModelVersionSpec(artifact_uri="s3://bucket/model.onnx")
        assert spec.trt_compile is None


# ---------------------------------------------------------------------------
# DEFAULT_TRT_IMAGE
# ---------------------------------------------------------------------------


class TestDefaultTrtImage:
    def test_uses_triton_version(self):
        from tsbk import DEFAULT_TRITON_VERSION

        assert DEFAULT_TRITON_VERSION in DEFAULT_TRT_IMAGE
        assert DEFAULT_TRT_IMAGE == f"nvcr.io/nvidia/tensorrt:{DEFAULT_TRITON_VERSION}-py3"
