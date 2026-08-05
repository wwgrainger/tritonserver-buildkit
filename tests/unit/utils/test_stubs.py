from unittest.mock import patch

from tsbk.utils.stubs import build_triton_stub


@patch("tsbk.utils.stubs._build_triton_stub")
def test_cache_bust_changes_local_and_shared_s3_stub_paths(mock_build, tmp_path, monkeypatch):
    monkeypatch.setattr("tsbk.utils.stubs.TSBK_DIR", tmp_path)
    monkeypatch.setattr("tsbk.utils.stubs.TSBK_S3_PREFIX", "s3://bucket/cache")

    first = build_triton_stub("3.12", "r26.03", cache_bust="release/one")
    second = build_triton_stub("3.12", "r26.03", cache_bust="release/two")

    assert first != second
    assert "release" not in first.name
    assert mock_build.call_count == 2
    first_s3_path = mock_build.call_args_list[0].kwargs["k8s_shared_s3_path"]
    second_s3_path = mock_build.call_args_list[1].kwargs["k8s_shared_s3_path"]
    assert first_s3_path != second_s3_path
    assert first_s3_path.endswith(first.name)
    assert second_s3_path.endswith(second.name)
