import subprocess
from pathlib import Path

import boto3
import mlflow
import numpy as np
import onnx
import pandas as pd
import pytest
import torch
import torch.onnx
from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import Schema, TensorSpec


@pytest.fixture(scope="session")
def assets_dir():
    return Path(__file__).parents[1] / "assets"
