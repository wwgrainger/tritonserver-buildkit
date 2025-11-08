
import boto3
import torch

s3c = boto3.client(
    "s3",
    region_name="us-east-2",
    endpoint_url="http://localhost:4566",
)

# Setup S3 bucket
bucket_name = "test-bucket"

s3_buckets = s3c.list_buckets()
if bucket_name not in [b["Name"] for b in s3_buckets["Buckets"]]:
    s3c.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": "us-east-2"},
    )

print(f"S3 bucket '{bucket_name}' is ready.")


# Create and upload Onnx model
class AddSubModel(torch.nn.Module):
    def forward(self, a, b):
        return a + b, a - b

model = AddSubModel()
model_path = "model.onnx"
dummy_input = (torch.randn(1, 3), torch.randn(1, 3))
torch.onnx.export(model, dummy_input, model_path, input_names=['a', 'b'], output_names=['add', 'sub'])

s3_key = "onnx/model.onnx"
s3c.upload_file(
    Filename=model_path,
    Bucket=bucket_name,
    Key=s3_key,
)

print(f"Model uploaded to s3://{bucket_name}/{s3_key}")