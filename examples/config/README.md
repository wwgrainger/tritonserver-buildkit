# Config Based Deployments

`tsbk` (Triton Server Build Kit) allows you to define and deploy Triton Inference Server model repositories using simple YAML configuration files. 
This example demonstrates an example configuration and how to use `tsbk` to build, run, and test models based on that configuration.

## Prerequisites

- Install example requirements:

```bash
pip install -r requirements.txt
```

- Start a localstack S3 service for model storage:

```bash
docker-compose up -d
```

## Setup

Create a model artifact and upload it to the localstack S3 service:

```bash
python create-model.py
```

## Run and Test with tsbk

We can use `tsbk run` to build, run, and test the model repository defined in `example-repo.yaml`:

```bash
AWS_S3_ENDPOINT_URL=http://localhost:4566 \
  tsbk run example-repo.yaml ./model-repo --test
```

This command does the following:
1. Builds the model repository in `./model-repo` based on the configuration in `example-repo.yaml`.
2. Launches Triton Inference Server in a Docker container, pointing to the built model repository.
3. Runs the defined test cases against the deployed models to verify functionality

If you want to run the server without testing, you can omit the `--test` flag:

```bash
AWS_ENDPOINT_URL_S3=http://localhost:4566 \
  tsbk run example-repo.yaml ./model-repo
```

This command will build the model repository and start the Triton server without executing tests.

## (Optional) Build Only

If you only want to build the model repository without running the server, use the `tsbk build` command:

```bash
AWS_ENDPOINT_URL_S3=http://localhost:4566 \
  tsbk build example-repo.yaml ./model-repo
```

This will create the Triton-compatible model repository structure in `./model-repo` based on the provided configuration.
You can then start the Triton server manually, pointing it to this repository. This also becomes an artifact you can version control or share.

## (Optional) Test Only

To run tests against an already deployed Triton server, use the `tsbk test` command:

```bash
AWS_ENDPOINT_URL_S3=http://localhost:4566 \
  tsbk test example-repo.yaml ./model-repo --url http://localhost:8000
```

This command will execute all defined test cases in `example-repo.yaml` against the Triton server running at the specified URL.

## Cleanup

After you are done, you can stop the localstack service:

```bash
docker-compose down
```



