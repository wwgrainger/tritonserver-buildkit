import os
from pathlib import Path
from typing import Tuple, Union

import boto3
import click

from tsbk import TSBK_DIR
from tsbk.utils import link_or_copy

s3r = boto3.resource("s3")
s3c = boto3.client("s3")


def split_s3_path(s3_path: str) -> Tuple[str, str]:
    """Splits an S3 URI into its bucket and key parts"""
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid s3 path: {s3_path}")

    path_parts = s3_path.replace("s3://", "").split("/")
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)
    return bucket, key


def s3_path_exists(s3_path: str) -> bool:
    bucket, key = split_s3_path(s3_path)
    res = s3c.list_objects(Bucket=bucket, Prefix=key)
    if "Contents" in res:
        return True
    return False


def compute_cache_path(s3_path: str) -> Path:
    """Helper function for computing a local path for a cached object.
    This is done by joining the s3 bucket and key with the ARTIFACT_CACHE_DIR

    Args:
        s3_path: the path of a s3 object

    Returns:
        a local path that can be used to cache the object
    """
    bucket, key = split_s3_path(s3_path)
    info = s3c.head_object(Bucket=bucket, Key=key)
    etag = info["ETag"].strip('"')
    return TSBK_DIR / "cache" / "s3" / bucket / f"{key}.{etag}"


def download_s3_file(origin_path: str, dst_path: Union[str, Path]):
    """Downloads a file from s3 to a cache path and creates a symlink to this path at the dst_path for Trition to use when serving models that use this S3 artifact

    Args:
        origin_path: the path of the file to download ex: s3://my-bucket/my.file
        dst_path: the local path to download the file to
    """
    bucket, key = split_s3_path(origin_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    local_cache_path = compute_cache_path(origin_path)
    if not local_cache_path.exists():
        local_cache_path.parent.mkdir(parents=True, exist_ok=True)
        click.secho(f"Downloading: s3://{bucket}/{key} -> {local_cache_path} (cache miss)", fg="blue")
        s3c.download_file(bucket, key, local_cache_path)
    click.secho(f"Creating hard link: {dst_path} -> {local_cache_path}", fg="blue")
    # rel_path = os.path.relpath(local_cache_path, dst_path.parent)
    link_or_copy(local_cache_path, dst_path)


def download_s3_path(origin_path: str, dst_path: Union[str, Path], **kwargs):
    """Downloads either a file or entire folder from s3 to a local caching path and creates symlinks in dst_path for each of these artifacts

    Args:
        origin_path: an S3 URI to either a file or folder
        dst_path: the path to download the file/folder to
    """

    bucket_name, prefix = split_s3_path(origin_path)
    bucket = s3r.Bucket(bucket_name)

    objects = list(bucket.objects.filter(Prefix=prefix))
    if len(objects) == 0:
        raise ValueError(f"No objects found at {origin_path}")

    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith("/"):  # Skip directories
            continue

        # Construct the local file path
        obj_local_path = os.path.join(dst_path, os.path.relpath(obj.key, prefix))

        # Download the file
        download_s3_file(f"s3://{bucket_name}/{obj.key}", obj_local_path)


def upload_s3_file(local_path: Union[str, Path], s3_path: str):
    """Upload a file to an S3 bucket

    Args:
        local_path: File to upload
        s3_path: the s3 location to copy the file to
    """
    bucket, key = split_s3_path(s3_path)
    click.secho(f"Uploading: {local_path} -> s3://{bucket}/{key}", fg="blue")
    s3c.upload_file(local_path, bucket, key)
