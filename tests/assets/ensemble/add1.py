import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []

        for request in requests:
            a = pb_utils.get_input_tensor_by_name(request, "a")

            b = a.as_numpy() + 1.0

            b = pb_utils.Tensor("b", b.astype(np.float32))

            inference_response = pb_utils.InferenceResponse(output_tensors=[b])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print("Cleaning up...")
