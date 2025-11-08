# Triton Server BuildKit (TSBK)

A Python toolkit that simplifies building, deploying, and testing NVIDIA Triton Inference Server model repositories.

## Overview

TSBK eliminates the manual complexity of configuring Triton model repositories by providing a declarative, YAML-based approach to model deployment. It handles everything from artifact management and environment setup to automated testing and Docker orchestration.

## Features

- **Multi-Backend Support**: Deploy models using Python, PyTorch, ONNX, TensorFlow, TensorRT, and MLflow backends
- **Flexible Artifact Management**: Pull models from S3, MLflow, or Databricks Unity Catalog
- **Automated Environment Setup**: Create and package conda environments with model dependencies
- **Ensemble Models**: Configure multi-model pipelines with dependency resolution
- **Built-in Testing**: Comprehensive testing framework with HTTP/gRPC client support
- **Docker Integration**: Automatic Triton server container management
- **Version Management**: Support multiple model versions with configurable version policies
- **Test Serialization**: Export test plans for CI/CD pipelines

## Prerequisites

- Python >= 3.11.0
- Poetry >= 1.8.3
- Docker (for running Triton server)
- NVIDIA Triton Inference Server container (pulled automatically when using `tsbk run`)

## Installation

### From PyPI (when published)

```bash
pip install tsbk
```

### From Source

```bash
git clone https://github.com/your-org/tritonserver-buildkit.git
cd tritonserver-buildkit
poetry install
```

## Quick Start

### 1. Create a Model Repository Configuration

Create a `model-repo.yaml` file defining your models:

```yaml
models:
  - name: my_model
    backend: python
    max_batch_size: 8
    versions:
      - version: 1
        source:
          type: s3
          uri: s3://my-bucket/models/my_model/
        conda:
          dependencies:
            - numpy
            - pandas
        test_cases:
          - name: basic_inference
            inputs:
              - name: input_data
                data: [[1.0, 2.0, 3.0]]
                dtype: FP32
                shape: [1, 3]
            outputs:
              - name: output_data
                data: [[4.0, 5.0, 6.0]]
                dtype: FP32
                shape: [1, 3]
```

### 2. Build the Model Repository

```bash
tsbk build --spec model-repo.yaml --output ./model_repository
```

This creates a Triton-compatible model repository structure with all necessary configuration files and dependencies.

### 3. Run Triton Server with Your Models

```bash
tsbk run --spec model-repo.yaml --output ./model_repository
```

This builds the repository and launches Triton server in a Docker container.

### 4. Test Your Models

```bash
tsbk test --spec model-repo.yaml --url http://localhost:8000
```

Runs all defined test cases against the deployed models.

## CLI Commands

### `tsbk version`
Display the installed version of TSBK.

### `tsbk build`
Build a Triton model repository from a configuration file.

**Options:**
- `--spec`: Path to model repository YAML specification
- `--output`: Directory where model repository will be created

### `tsbk run`
Build and run Triton server with your models in Docker.

**Options:**
- `--spec`: Path to model repository YAML specification
- `--output`: Directory for model repository
- `--image`: Triton server Docker image (default: auto-detected)
- `--gpus`: GPU configuration (e.g., `all`, `device=0,1`)
- Additional Docker run options

### `tsbk test`
Test models against a running Triton server.

**Options:**
- `--spec`: Path to model repository YAML specification
- `--url`: Triton server URL (default: http://localhost:8000)
- `--protocol`: Communication protocol (`http` or `grpc`)

### `tsbk create-test-plan`
Serialize test plan to msgpack format for CI/CD pipelines.

**Options:**
- `--spec`: Path to model repository YAML specification
- `--output`: Output file path

### `tsbk run-test-plan`
Execute a serialized test plan.

**Options:**
- `--test-plan`: Path to msgpack test plan file
- `--url`: Triton server URL
- `--protocol`: Communication protocol

## Configuration

### Model Repository Specification

The YAML configuration defines your entire model repository:

```yaml
models:
  - name: model_name
    backend: python|pytorch|onnxruntime|tensorflow|tensorrt|mlflow
    max_batch_size: 8
    instance_group:
      - count: 1
        kind: KIND_GPU
    versions:
      - version: 1
        source:
          type: s3|mlflow|databricks
          uri: s3://bucket/path or models:/model-name/version
        conda:
          channels:
            - conda-forge
          dependencies:
            - numpy>=1.20
            - pandas
        test_cases:
          - name: test_case_name
            inputs:
              - name: input_name
                data: [[1.0, 2.0]]
                dtype: FP32
                shape: [1, 2]
            outputs:
              - name: output_name
                data: [[3.0, 4.0]]
                dtype: FP32
                shape: [1, 2]
                rtol: 1e-5  # Relative tolerance
                atol: 1e-8  # Absolute tolerance
```

### Supported Backends

- **python**: Custom Python code with triton_python_backend
- **pytorch**: PyTorch models (TorchScript)
- **onnxruntime**: ONNX models
- **tensorflow**: TensorFlow SavedModel
- **tensorrt**: TensorRT engines
- **mlflow**: MLflow models with auto-detected backend

### Source Types

- **s3**: Download from AWS S3 or S3-compatible storage
- **mlflow**: Pull from MLflow Model Registry
- **databricks**: Download from Databricks Unity Catalog

## Development

### Setup Development Environment

```bash
poetry install --with dev
pre-commit install
```

### Run Tests

**Unit Tests:**
```bash
make unit-tests
```

**Integration Tests:**
```bash
make integration-tests
```

Integration tests use LocalStack to simulate AWS S3 locally.

### Code Quality

This project uses:
- **black**: Code formatting
- **isort**: Import sorting
- **pre-commit**: Automated code quality checks

Run checks manually:
```bash
pre-commit run --all-files
```

## Examples

See `tests/assets/example-repo.yaml` for a complete example with:
- Python backend models
- Ensemble models
- Multiple versions
- Comprehensive test cases

## CI/CD Integration

Export test plans for your CI pipeline:

```bash
# Create test plan
tsbk create-test-plan --spec model-repo.yaml --output test_plan.msgpack

# In CI: run test plan against deployed server
tsbk run-test-plan --test-plan test_plan.msgpack --url http://triton-server:8000
```

## Troubleshooting

### Docker Issues

If `tsbk run` fails to start Triton:
- Ensure Docker daemon is running
- Check you have permissions to run Docker commands
- Verify the Triton image is accessible

### Model Loading Failures

Check:
- Model artifacts are correctly downloaded to version directories
- `config.pbtxt` is valid (use `tsbk build` first to validate)
- Dependencies are properly specified in conda configuration
- Backend is correctly specified for your model format

### Test Failures

- Verify Triton server is running and accessible
- Check test case input/output shapes match model signature
- Adjust tolerance values (`rtol`, `atol`) for numerical comparisons

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass and pre-commit checks succeed
6. Submit a pull request

## License

See LICENSE file for details.

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation and examples

## Acknowledgments

Built by the Grainger MLOps & Platform Team for simplifying Triton Inference Server deployments.
