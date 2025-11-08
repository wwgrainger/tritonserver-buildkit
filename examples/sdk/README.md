# Config Based Deployments

`tsbk` (Triton Server Build Kit) has an expressive SDK that allows you to define Triton model repositories in python code and build, run, and test them with easy to use functions.

## Prerequisites

- Install example requirements:

```bash
pip install -r requirements.txt
```

## Setup

Create a model artifact and register it with mlflow.

```bash
python create-model.py
```

## Run and Test with python

In this folder is a `server.py` that demonstrates how to use the SDK to build, run, and test a triton model repository all in code.

```bash
python server.py --test
```

This command will build the model repository, launch triton server in a docker container, and run the mlflow registered input example as a test case against the deployed model.
The `--test` flag is optional, if not provided the script will build and run the model repository.

## (Optional) Build and Run Separately

You can also build and run the model repository separately if desired.

```bash
python server.py --build-only
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/model-repo:/models \
  nvcr.io/nvidia/tritonserver:25.08-py3 \
  tritonserver --model-repository=/models
```
