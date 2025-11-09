import argparse

import tsbk


def model_repo(model_repo_path: str, artifact_uri: str) -> tsbk.TritonModelRepo:
    return tsbk.TritonModelRepo(
        "example-server",
        path=model_repo_path,
        models={"addsub": tsbk.TritonModel(versions=[tsbk.TritonModelVersion(artifact_uri=artifact_uri)])},
    )


def main(args):
    repo = model_repo(args.model_repo, args.model_artifact_uri)
    repo.build()

    if args.build_only:
        return

    repo.run(detach=args.test)

    if args.test:
        repo.test(url=repo.http_url)
        repo.test(url=repo.grpc_url, grpc=True)
        repo.stop()
        print("Tests passed!")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage Triton Inference Server with tsbk")
    parser.add_argument("--model_artifact_uri", type=str, help="Model artifact to deploy", default="models:/addsub/1")
    parser.add_argument("--model-repo", type=str, default="./model-repo", help="Path to the model repository")
    parser.add_argument(
        "--build-only", action="store_true", help="Only build the model repository without starting the server"
    )
    parser.add_argument("--test", action="store_true", help="Run in test mode without starting the server")
    args = parser.parse_args()

    assert not (args.build_only and args.test), "Cannot use --build-only and --test together"

    main(args)
