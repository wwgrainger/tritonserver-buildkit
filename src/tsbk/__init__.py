import os
from pathlib import Path

__version__ = "1.8.0"

# env vars
TSBK_DIR = Path(os.environ.get("TSBK_DIR", "./.tsbk"))
TSBK_S3_PREFIX = os.environ.get("TSBK_S3_PREFIX")
TSBK_K8S_SERVICE_ACCOUNT = os.environ.get("TSBK_K8S_SERVICE_ACCOUNT", "default")

# constants
DEFAULT_TRITON_VERSION = "25.08"

from .cli import tsbk
from .model import TritonModel
from .model_repo import TritonModelRepo
from .model_version import TritonModelVersion
from .test_case import TestCase
from .triton_grpc_model_client import TritonGrpcModelClient
from .triton_http_model_client import TritonHttpModelClient
from .types import TritonDType
