# tsbk - Triton Server Build Kit

A powerful Python toolkit for building, deploying, and testing [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) model repositories with ease.

## Overview

`tsbk` (Triton Server Build Kit) simplifies the process of creating and managing Triton Inference Server deployments. It provides both a declarative YAML-based configuration system and a programmatic Python SDK, making it easy to:

- Build Triton-compatible model repositories from simple configurations
- Deploy and run Triton servers in Docker containers
- Test model deployments with built-in validation framework
- Integrate with MLflow for model management

Whether you're developing ML models locally or deploying them to production, `tsbk` streamlines the entire workflow.

## Features

- **Declarative Configuration**: Define model repositories using simple YAML files
- **Programmatic SDK**: Build model repositories programmatically using Python
- **Automatic Testing**: Built-in test framework with input/output validation
- **MLflow Integration**: Seamlessly load models from MLflow registry
- **S3 Support**: Fetch model artifacts from S3-compatible storage
- **Multi-Backend Support**: ONNX, TensorRT, Python, ensemble models, and more
- **Docker Integration**: Automatic Triton server deployment in containers
- **Test Plan Serialization**: Create reusable test plans for CI/CD pipelines
- **HTTP & gRPC Support**: Test models via both protocols

## Installation

### Using pip

```bash
pip install tsbk
```

## Requirements

- Python 3.11+
- Docker (for running Triton servers)
- AWS credentials (for S3 model storage, optional)
- Databricks credentials (for MLflow model access, optional)

## Quick Start

### CLI: YAML Configuration

1. **Create a model configuration** (`model-config.yaml`):

```yaml
name: quickstart
models:
  your-model-name:
    backend: onnxruntime
    versions:
      - artifact_uri: s3://your-bucket/model.onnx
```

2. **Build and run** with a single command:

```bash
tsbk run model-config.yaml ./model-repo
```

This will:
- Build the tritonserver model repository at `./model-repo`
- Launch Triton server in a Docker container

### SDK: Python API

```python
import tsbk

# Define your model repository
repo = tsbk.TritonModelRepo(
    name='quickstart',
    path='./model-repo',
    models={
        'your-model-name': tsbk.TritonModel(
            backend='onnxruntime',
            versions=[
                tsbk.TritonModelVersion(
                    artifact_uri='models:/your-onnx-model/1'  # MLflow model URI or S3 URI supported
                )
            ]
        )
    }
)

# Build the repository
repo.build()

# Run Triton server
repo.run()
```

## CLI Commands

### `tsbk build`
Build a Triton model repository from a configuration file.

```bash
tsbk build model-config.yaml ./model-repo
```

### `tsbk run`
Build and run a Triton server from a configuration file.

```bash
# Run with defaults
tsbk run model-config.yaml ./model-repo
```

Optionally detach from the server to run in the background:

```bash
# Detach mode
tsbk run model-config.yaml ./model-repo --detach
```

Optionally run tests after starting the server:

```bash
# Run and test
tsbk run model-config.yaml ./model-repo --test
```

### `tsbk test`
Test a running Triton server against your configuration. This requires providing test cases for your models.
They can be specified via YAML or programmatically in the SDK. `tsbk` also supports taking test data from MLFlow models with defined `example_inputs`.

```bash
# Test HTTP endpoint
tsbk test model-config.yaml ./model-repo --url http://localhost:8000

# Test gRPC endpoint
tsbk test model-config.yaml ./model-repo --url localhost:8001 --grpc
```

### `tsbk create-test-plan`
Create a serialized test plan for CI/CD pipelines.

```bash
tsbk create-test-plan model-config.yaml ./model-repo test-plan.msgpack
```

### `tsbk run-test-plan`
Execute a serialized test plan against a running server.

```bash
tsbk run-test-plan test-plan.msgpack --url http://localhost:8000
```

## Examples

Explore the [`examples/`](examples/) directory for complete working examples:

- **[Config-based deployment](examples/config/)**: Using YAML configuration files
- **[SDK-based deployment](examples/sdk/)**: Using the Python SDK programmatically

## Environment Variables

- `TSBK_DIR`: Working directory for tsbk operations (default: `./.tsbk`)
- `TSBK_S3_PREFIX`: S3 prefix for temporary shared model artifacts
- `TSBK_K8S_SERVICE_ACCOUNT`: Kubernetes service account used when running build jobs (default: `default`)

## Troubleshooting

### Docker Issues

If Triton server fails to start:
- Ensure Docker is running
- Check port availability (8000, 8001, 8002)
- Verify Docker has sufficient resources (memory, disk space)

### Model Loading Errors

If models fail to load:
- Verify artifact URIs are correct
- Ensure model format matches backend (e.g., `.onnx` for onnxruntime)
- Review Triton server logs: `docker logs <container-id>`

### Test Failures

If tests fail unexpectedly:
- Check tolerance settings (`rtol`, `atol`)
- Verify input/output tensor shapes match model expectations
- Review expected output values

## Acknowledgments

- Built on [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- Integrates with [MLflow](https://mlflow.org/) for model management

## Roadmap

Future enhancements we're considering:
- Kubernetes deployment support
- TensorRT model optimization utilities
- Performance benchmarking tools
- Direct Hugging Face model support

Suggestions and contributions are welcome!
