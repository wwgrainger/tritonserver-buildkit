from tsbk import TritonModelRepo
from tsbk.spec import (
    NumpyArraySpec,
    TestCaseSpec,
    TritonDTypeSpec,
    TritonModelRepoSpec,
    TritonModelSpec,
    TritonModelVersionSpec,
)


def test_spec_to_repo(tmp_path, s3_pytorch_model):
    spec = TritonModelRepoSpec(
        name="test_model_repo",
        models={
            "pytorch-model": TritonModelSpec(
                backend="pytorch",
                inputs=[
                    TritonDTypeSpec(name="a", dtype="float32", dims=[-1]),
                    TritonDTypeSpec(name="b", dtype="float32", dims=[-1]),
                ],
                outputs=[
                    TritonDTypeSpec(name="add", dtype="float32", dims=[-1]),
                    TritonDTypeSpec(name="sub", dtype="float32", dims=[-1]),
                ],
                versions=[TritonModelVersionSpec(artifact_uri=s3_pytorch_model)],
                test_cases=[
                    TestCaseSpec(
                        inputs={
                            "a": NumpyArraySpec(
                                dtype="float32",
                                dims=[3],
                                data=[1.0, 2.0, 3.0],
                            ),
                            "b": NumpyArraySpec(
                                dtype="float32",
                                dims=[3],
                                data=[4.0, 5.0, 6.0],
                            ),
                        },
                        expected_outputs={
                            "add": NumpyArraySpec(
                                dtype="float32",
                                dims=[3],
                                data=[5.0, 7.0, 9.0],
                            ),
                            "sub": NumpyArraySpec(
                                dtype="float32",
                                dims=[3],
                                data=[-3.0, -3.0, -3.0],
                            ),
                        },
                    )
                ],
            )
        },
        triton_image="some-image",
    )

    TritonModelRepo(path=tmp_path, **spec.model_dump(mode="json"))
