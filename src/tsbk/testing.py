from dataclasses import dataclass

from tsbk.test_case import TestCase, TestResult
from tsbk.triton_grpc_model_client import TritonGrpcModelClient
from tsbk.triton_http_model_client import TritonHttpModelClient


@dataclass
class TritonModelVersionTestPlan:
    name: str
    version: int
    test_cases: list[TestCase]
    decoupled_transaction_policy: bool

    def run_tests(
        self, url: str, ca_certs: str | None = None, headers: dict | None = None, grpc: bool = False
    ) -> list[TestResult]:
        """Runs the test cases defined for this model version.

        Args:
            url: The URL of the Triton server to test against.
            ca_certs: Path to the CA certificates file for secure connections, if applicable.
            headers: Optional headers to include in the request.
            grpc: If True, use gRPC client, otherwise use HTTP client.

        Returns:
            A list of TestResult objects containing the results of the tests.
        """
        if not self.test_cases:
            return [
                TestResult(
                    success=False,
                    model_name=self.name,
                    model_version=self.version,
                    message=f"No test cases defined for {self.name}:{self.version}",
                    missing_test=True,
                )
            ]

        if self.decoupled_transaction_policy and not grpc:
            return [
                TestResult(
                    success=False,
                    model_name=self.name,
                    model_version=self.version,
                    message=f"Decoupled transaction policy is not supported for HTTP client. Use gRPC client for {self.name}:{self.version}",
                    decoupled_with_http=True,
                )
            ]

        kwargs = {
            "url": url,
            "headers": headers,
            "ca_certs": ca_certs,
            "model_name": self.name,
            "model_version": self.version,
        }

        model_client = TritonGrpcModelClient(**kwargs) if grpc else TritonHttpModelClient(**kwargs)

        results = list()
        for test_case in self.test_cases:
            print(f"Running {'grpc' if grpc else 'http'} test for {self.name}:{self.version} - ", end="")
            result = test_case.run(model_client, decoupled=self.decoupled_transaction_policy)
            print("✅" if result.success else "❌")
            results.append(result)
        return results

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "decoupled_transaction_policy": self.decoupled_transaction_policy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TritonModelVersionTestPlan":
        from tsbk.test_case import TestCase

        return cls(
            name=d["name"],
            version=d["version"],
            test_cases=[TestCase.from_dict(tc) for tc in d["test_cases"]],
            decoupled_transaction_policy=d["decoupled_transaction_policy"],
        )


@dataclass
class TritonModelTestPlan:
    version_plans: list[TritonModelVersionTestPlan]
    ensemble_models: list[tuple[str, int]] | None

    def run_tests(
        self, url: str, ca_certs: str | None = None, headers: dict | None = None, grpc: bool = False
    ) -> list[TestResult]:
        """Runs the test cases defined for this model version.

        Args:
            url: The URL of the Triton server to test against.
            ca_certs: Path to the CA certificates file for secure connections, if applicable.
            headers: Optional headers to include in the request.
            grpc: If True, use gRPC client, otherwise use HTTP client.

        Returns:
            A list of TestResult objects containing the results of the tests.
        """
        results = list()
        for version_plan in self.version_plans:
            results.extend(version_plan.run_tests(url, ca_certs=ca_certs, headers=headers, grpc=grpc))
        return results

    def to_dict(self) -> dict:
        return {
            "version_plans": [vp.to_dict() for vp in self.version_plans],
            "ensemble_models": self.ensemble_models,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TritonModelTestPlan":
        return cls(
            version_plans=[TritonModelVersionTestPlan.from_dict(vp) for vp in d["version_plans"]],
            ensemble_models=[tuple(model) for model in d["ensemble_models"] or []],  # noqa
        )


@dataclass
class TritonModelRepoTestPlan:
    model_plans: list[TritonModelTestPlan]
    repo_models: list[tuple[str, int]]

    def run_tests(
        self, url: str, ca_certs: str | None = None, headers: dict | None = None, grpc: bool = False
    ) -> list[TestResult]:
        """Runs the test cases defined for all models in this repository against the Triton server.

        Args:
            url: The URL of the Triton server to test against.
            ca_certs: Path to CA certificates for secure connections, if applicable.
            headers: Additional headers to include in the requests.
            grpc: Whether to use gRPC for testing.

        Returns:
            A list of TestResult objects containing the results of the tests.
        """
        self.check_server_matches(url, ca_certs=ca_certs, headers=headers, grpc=grpc)
        results: list[TestResult] = []
        for model in self.model_plans:
            results.extend(model.run_tests(url, ca_certs=ca_certs, headers=headers, grpc=grpc))

        for result in results:
            if result.missing_test and self._test_result_covered_by_ensemble(result):
                print(f"Testing {result.model_name}:{result.model_version} are covered by an ensemble model ✅")
                result.success = True
            elif result.missing_test:
                print(f"No test cases defined for {result.model_name}:{result.model_version} ❌")
            elif not result.success:
                print(f"Failure details for {result.model_name}/{result.model_version}")
                if result.message:
                    print(f"Message: {result.message}")
                if result.exception:
                    print(f"Exception: {result.exception}")
                if result.traceback:
                    print(f"Traceback: \n{result.traceback}")
                print("")
        return results

    def check_server_matches(
        self, url: str, ca_certs: str | None = None, headers: dict | None = None, grpc: bool = False
    ):
        """Checks that the server has the same models as the repo."""
        server_models = set(self.get_server_models(url, ca_certs=ca_certs, headers=headers, grpc=grpc))
        repo_models = set(self.repo_models)
        if server_models != repo_models:
            raise ValueError(
                f"Server models do not match repo models. Missing models: {repo_models - server_models}, Unexpected models: {server_models - repo_models}"
            )

    @staticmethod
    def get_server_models(
        url: str, ca_certs: str | None = None, headers: dict | None = None, grpc: bool = False
    ) -> list[tuple[str, int]]:
        """Returns a list of models that are currently loaded in the server"""
        if not grpc:
            triton_client = TritonHttpModelClient.create_client(url, ca_certs=ca_certs)
            model_index = triton_client.get_model_repository_index(headers=headers)
            return [(model["name"], int(model["version"])) for model in model_index]
        else:
            triton_client = TritonGrpcModelClient.create_client(url, ca_certs=ca_certs)
            model_index = triton_client.get_model_repository_index(headers=headers)
            return [(model.name, int(model.version)) for model in model_index.models]

    def _test_result_covered_by_ensemble(self, result: TestResult) -> bool:
        """Checks if a test result is covered by an ensemble model."""
        for model in self.model_plans:
            if model.ensemble_models:
                for model_name, model_version in model.ensemble_models:
                    if result.model_name == model_name and result.model_version == model_version:
                        return True
        return False

    def to_dict(self) -> dict:
        return {
            "model_plans": [mp.to_dict() for mp in self.model_plans],
            "repo_models": self.repo_models,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TritonModelRepoTestPlan":
        return cls(
            model_plans=[TritonModelTestPlan.from_dict(mp) for mp in d["model_plans"]],
            repo_models=[tuple(model) for model in d["repo_models"]],  # noqa
        )
