from unittest.mock import patch

from tsbk.model_version import TritonModelVersion


@patch("tsbk.model_version.link_or_copy")
@patch("tsbk.model_version.build_trt_engine")
@patch("tsbk.model_version._find_onnx_file")
@patch("tsbk.model_version.download_s3_path")
def test_tensor_rt_build_forwards_gpu_toleration_key(
    mock_download,
    mock_find_onnx,
    mock_build_engine,
    mock_link_or_copy,
    tmp_path,
):
    onnx_path = tmp_path / "source" / "model.onnx"
    plan_path = tmp_path / "engine.plan"
    mock_find_onnx.return_value = onnx_path
    mock_build_engine.return_value = plan_path
    model_version = TritonModelVersion(
        artifact_uri="s3://bucket/model.onnx",
        trt_compile={"enabled": True, "gpu_toleration_key": "nvidia.com/gpu"},
    )
    model_version.path = tmp_path / "model" / "1"
    model_version.backend = "tensorrt"

    model_version.build()

    mock_build_engine.assert_called_once_with(
        onnx_path=onnx_path,
        precision=None,
        workspace_size=None,
        extra_args=None,
        trt_image=None,
        gpu_name=None,
        instance_family=None,
        gpu_toleration_key="nvidia.com/gpu",
        cache_bust=None,
    )
    mock_link_or_copy.assert_called_once_with(plan_path, model_version.path / "model.plan")
