import pytest

from tsbk.utils.conda import calc_hash


@pytest.fixture()
def requirements1(tmp_path):
    req = tmp_path.joinpath("requirements1.txt")
    req.write_text("req1==1\nreq2==2")
    return req


@pytest.fixture()
def requirements2(tmp_path):
    req = tmp_path.joinpath("requirements2.txt")
    req.write_text("req3==1\nreq4==2")
    return req


def test_calc_hash():
    assert (
        calc_hash(b"my\nrequirements", b"my_platform", "3.10")
        == "e70fa16d6309a15005ee29f9b32fd9011f56128f39b0f12fb10f88375b0c2b15"
    )

    assert calc_hash(b"my\nrequirements", b"my_platform", "3.10") != calc_hash(
        b"my\nother\nrequirements", b"my_platform", "3.10"
    )
    assert calc_hash(b"my\nrequirements", b"my_platform", "3.10") != calc_hash(
        b"my\nrequirements", b"my_other_platform", "3.10"
    )
    assert calc_hash(b"my\nrequirements", b"my_platform", "3.10") != calc_hash(
        b"my\nrequirements", b"my_platform", "3.11"
    )
