from pydantic import BaseModel, ConfigDict

from tsbk import DEFAULT_TRITON_VERSION


class NumpyArraySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dtype: str
    """The data type of the NumPy array, which can be a string representation of the NumPy data type (e.g., 'float32', 'int64')."""
    dims: list[int]
    """The dimensions of the NumPy array, which is a list of integers representing the shape of the array."""
    data: list
    """The data of the NumPy array, which is a list of values that can be converted to the specified data type."""


class TestCaseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: dict[str, NumpyArraySpec]
    """The input data for the test case, which is a dictionary mapping input names to lists of values."""
    expected_outputs: dict[str, NumpyArraySpec]
    """The expected output data for the test case, which is a dictionary mapping output names to lists of values."""
    allow_nan: bool = False
    """Whether to allow NaN values in the output. If True, NaN values will not raise an error during validation."""
    allow_inf: bool = False
    """Whether to allow Inf values in the output. If True, Inf values will not raise an error during validation."""
    rtol: float = 1.0e-5
    """The relative tolerance to use for comparing floating point values. This is used to determine if two floating point values are close enough to be considered equal."""
    atol: float = 1.0e-7
    """The absolute tolerance to use for comparing floating point values. This is used to determine if two floating point values are close enough to be considered equal."""


class TritonDTypeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    """The name of the Triton data type, which can be used to specify the input"""
    dtype: str
    """The Triton data type, which can be one of the supported Triton data types."""
    dims: list[int]
    """The dimensions of the Triton data type, which can be used to specify the shape of the input or output tensors."""


class TritonModelVersionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_uri: str | None = None
    """The URI of the model artifact, which can be an MLflow model or an S3 object."""
    python_model_file: str | None = None
    """The path to the Python model file, which is required for Python models."""
    version: int | None = None
    """The version number of the model"""
    test_cases: list[TestCaseSpec] | None = None
    """A list of test cases for the model version, which can be used to validate the model's behavior."""


class TritonModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    versions: list[TritonModelVersionSpec] | None = None
    """A dictionary of model versions to deploy (optional, not required for ensemble models)"""
    backend: str | None = None
    """The Triton backend for this model to use, e.g. 'python', 'tensorrt', 'onnxruntime', etc."""
    platform: str | None = None
    """The tritonserver platform for this model to use, e.g. 'ensemble'"""
    max_batch_size: int | None = None
    """The maximum batch size for the model, which can be used to control the batching behavior"""
    inputs: list[TritonDTypeSpec] | None = None
    """A dictionary of input names to Triton data types, which can be used to specify the input tensors for the model."""
    outputs: list[TritonDTypeSpec] | None = None
    """A dictionary of output names to Triton data types, which can be used to specify the output tensors for the model."""
    config: str | None = None
    """Inline Config for Triton Model configuration"""
    config_file: str | None = None
    """ The path to the model's Triton configuration file (i.e. models/<model_name>/config.pbtxt)"""
    instance_group: list[dict] | None = None
    """The instance group configuration for the model, which can be used to specify the number of model instances and their placement."""
    python_version: str | None = None
    """ The python version to use for the model. """
    requirements_file: str | None = None
    """ The path to the model's requirements file. (i.e. models/<model_name>/requirements.txt)"""
    test_cases: list[TestCaseSpec] | None = None
    """A list of test cases for the model, which can be used to validate the model's behavior. Test cases supplied for a model are run for all model version."""


class TritonModelRepoSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    """The name of the Triton model repository, which is used as the directory name in the Triton model repository."""
    models: dict[str, TritonModelSpec]
    """A list of Triton models to deploy in the model repository."""
    triton_image: str = "nvcr.io/nvidia/tritonserver"
    """The Docker image to use for the Triton server."""
    triton_image_tag: str = f"{DEFAULT_TRITON_VERSION}-py3"
    """The Docker image tag to use for the Triton server."""
