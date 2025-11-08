import queue
from collections import defaultdict
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype, triton_to_np_dtype


class TritonGrpcModelClient:
    """A client for interacting with Triton models over HTTP"""

    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: int | str | None = None,
        ca_certs: str | None = None,
        headers: dict | None = None,
    ):
        """A client for interacting with Triton models over HTTP

        Args:
            url: The url of the Triton server
            model_name: The name of the model to interact with
            model_version: The version of the model to interact with
            ca_certs: The path to the CA certificates file
            headers: Headers to include with each request
        """
        self.url = url
        self.model_name = model_name
        self.model_version = "" if model_version is None else str(model_version)

        self.client = self.create_client(url, ca_certs=ca_certs)
        self._model_config = None
        self.headers = headers or {}

    @staticmethod
    def create_client(url: str, ca_certs: str | None = None) -> grpcclient.InferenceServerClient:
        if url.startswith("http://"):
            ssl = False
            host = url[len("http://") :]
        elif url.startswith("https://"):
            ssl = True
            host = url[len("https://") :]
        else:
            raise ValueError(f"URL must start with 'http://' or 'https://', got: {url}")

        kwargs = {"url": host, "ssl": ssl}
        if ssl and ca_certs:
            kwargs["root_certificates"] = ca_certs
        return grpcclient.InferenceServerClient(**kwargs)

    def infer(
        self, inputs: dict[str, np.ndarray], output_names: list[str] | None = None, headers: dict[str, str] | None = None
    ) -> dict[str, np.ndarray]:
        """Run inference using the model

        Args:
            inputs: The inputs to the model
            output_names: The names of the outputs to return
            headers: The headers to use for the request

        Returns:
            The outputs of the model
        """
        headers = {**self.headers, **(headers or {})}
        self._validate_inputs(inputs)
        self._validate_output_names(output_names)

        infer_inputs = []
        for input_name in inputs:
            input_dtype = self.input_np_type(input_name)
            infer_inputs.append(
                grpcclient.InferInput(input_name, list(inputs[input_name].shape), np_to_triton_dtype(input_dtype))
            )
            infer_inputs[-1].set_data_from_numpy(inputs[input_name].astype(input_dtype))

        output_names = output_names or [output["name"] for output in self.model_outputs]
        infer_outputs = []
        for output_name in output_names:
            infer_outputs.append(grpcclient.InferRequestedOutput(output_name))

        response = self.client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=infer_inputs,
            outputs=infer_outputs,
            headers=headers,
        )

        return {name: response.as_numpy(name) for name in output_names}

    def infer_decoupled(
        self, inputs: dict[str, np.ndarray], output_names: list[str] | None = None, headers: dict[str, str] | None = None
    ) -> dict[str, np.ndarray]:
        class UserData:
            def __init__(self):
                self.completed_requests = queue.Queue()

        def callback(user_data, result, error):
            if error:
                user_data.completed_requests.put(error)
            else:
                user_data.completed_requests.put(result)

        user_data = UserData()
        self.client.start_stream(callback=partial(callback, user_data), headers=headers)  # noqa
        try:
            grpc_inputs = list()
            for name, arr in inputs.items():
                grpc_input = grpcclient.InferInput(name, list(arr.shape), np_to_triton_dtype(arr.dtype))
                grpc_input.set_data_from_numpy(arr)
                grpc_inputs.append(grpc_input)

            output_names = output_names or [output["name"] for output in self.model_outputs]
            infer_outputs = []
            for output_name in output_names:
                infer_outputs.append(grpcclient.InferRequestedOutput(output_name))

            self.client.async_stream_infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=grpc_inputs,
                enable_empty_final_response=True,
                request_id=str(1),
            )

            outputs = defaultdict(list)
            while True:
                data_item = user_data.completed_requests.get()
                if isinstance(data_item, grpcclient.InferenceServerException):
                    raise data_item
                if data_item.get_response(as_json=True)["parameters"]["triton_final_response"]["bool_param"]:
                    return {name: np.array(outputs[name]) for name in outputs}

                output_names = [output["name"] for output in data_item.get_response(as_json=True)["outputs"]]

                for output_name in output_names:
                    outputs[str(output_name)].append(data_item.as_numpy(output_name))
        finally:
            self.client.stop_stream()

    @property
    def model_config(self) -> dict:
        """The model configuration"""
        if self._model_config is None:
            self._model_config = self.client.get_model_config(self.model_name, headers=self.headers, as_json=True)[
                "config"
            ]
        return self._model_config

    @property
    def model_inputs(self) -> list[dict]:
        """The model input configuration"""
        return self.model_config["input"]

    @property
    def model_outputs(self) -> list[dict]:
        """The model output configuration"""
        return self.model_config["output"]

    def input_np_type(self, input_name: str):
        for model_input in self.model_inputs:
            if model_input["name"] == input_name:
                triton_dtype = model_input["data_type"].split("_")[1]
                if triton_dtype == "STRING":
                    triton_dtype = "BYTES"
                return triton_to_np_dtype(triton_dtype)
        raise ValueError(f"Input {input_name} not found in model inputs")

    def _validate_inputs(self, inputs: dict[str, np.ndarray]):
        for input_name in inputs:
            if not any(input_name == input["name"] for input in self.model_inputs):
                raise ValueError(f"Input {input_name} is not part of the model inputs")

    def _validate_output_names(self, output_names: list[str] | None):
        if output_names is None:
            return
        for output_name in output_names:
            if not any(output_name == output["name"] for output in self.model_outputs):
                raise ValueError(f"Output {output_name} is not part of the model outputs")
