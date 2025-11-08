
import torch
import onnx
import mlflow
import numpy as np


# Create Onnx model
class AddSubModel(torch.nn.Module):
    def forward(self, a, b):
        return a + b, a - b

model = AddSubModel()
model_path = "model.onnx"
dummy_input = (torch.randn(1, 3), torch.randn(1, 3))
torch.onnx.export(model, dummy_input, model_path, input_names=['a', 'b'], output_names=['add', 'sub'])

onnx_model = onnx.load(model_path)

# Log model to MLflow
with mlflow.start_run() as run:
    model = mlflow.onnx.log_model(
        onnx_model,
        artifact_path="onnx_model",
        registered_model_name="addsub",
        input_example={
            "a": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "b": np.array([[4.0, 5.0, 6.0]], dtype=np.float32),
        },
    )