"""AWS client utilities for Another Automatic Video Editor."""

import boto3
from botocore.config import Config as BotoConfig


def boto3_config() -> BotoConfig:
    """Create boto3 config with retries for transient issues."""
    return BotoConfig(
        retries={"max_attempts": 10, "mode": "standard"},
        connect_timeout=10,
        read_timeout=300,
    )


def rekognition_client(region: str):
    """Create Rekognition client."""
    return boto3.client("rekognition", region_name=region, config=boto3_config())


def transcribe_client(region: str):
    """Create Transcribe client."""
    return boto3.client("transcribe", region_name=region, config=boto3_config())


def bedrock_runtime_client(region: str):
    """Create Bedrock Runtime client."""
    return boto3.client("bedrock-runtime", region_name=region, config=boto3_config())


def s3_client(region: str):
    """Create S3 client."""
    return boto3.client("s3", region_name=region, config=boto3_config())
