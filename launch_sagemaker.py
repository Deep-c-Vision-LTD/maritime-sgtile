"""
Launch a SageMaker training job for YOLO11s via boto3 (no SageMaker SDK needed).
Dataset already uploaded to S3 — run with --skip-upload.

Usage:
    python launch_sagemaker.py --skip-upload
"""

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import time

import boto3

# ── config ────────────────────────────────────────────────────────────────────
REGION        = "us-east-1"
ACCOUNT       = "YOUR_ACCOUNT_ID"
ROLE          = (f"arn:aws:iam::{ACCOUNT}:role/service-role/"
                 "YOUR_SAGEMAKER_ROLE")
S3_BUCKET     = f"sagemaker-us-east-1-{ACCOUNT}"
S3_PREFIX     = "maritime-sgtile"
INSTANCE_TYPE = "ml.g5.xlarge"   # A10G 24 GB — auto-terminates on completion
# PyTorch 2.3 + CUDA 12.1 DLC (script-mode)
IMAGE_URI     = (f"763104351884.dkr.ecr.{REGION}.amazonaws.com/"
                 "pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker")
LOCAL_TILES   = "data/work/tiles"
JOB_NAME      = f"yolo11s-sds-{int(time.time())}"
# ──────────────────────────────────────────────────────────────────────────────


def upload_dataset(skip=False):
    s3_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/tiles/"
    if skip:
        print(f"[upload] skipping — using {s3_uri}")
        return s3_uri
    print(f"[upload] syncing {LOCAL_TILES}/ → {s3_uri} (~6 GB) …")
    r = subprocess.run(["aws", "s3", "sync", LOCAL_TILES, s3_uri,
                        "--exclude", "*.cache", "--no-progress"])
    if r.returncode != 0:
        sys.exit("[upload] ERROR: aws s3 sync failed")
    print("[upload] done")
    return s3_uri


def upload_sourcecode():
    """Tar train_tiled.py + requirements.txt, upload to S3, return S3 URI."""
    s3 = boto3.client("s3", region_name=REGION)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        code_tar = tmp.name
    with tarfile.open(code_tar, "w:gz") as tar:
        tar.add("train/train_tiled.py",   arcname="train_tiled.py")
        tar.add("train/requirements.txt", arcname="requirements.txt")
    key = f"{S3_PREFIX}/code/{JOB_NAME}/sourcedir.tar.gz"
    s3.upload_file(code_tar, S3_BUCKET, key)
    os.unlink(code_tar)
    uri = f"s3://{S3_BUCKET}/{key}"
    print(f"[code] uploaded → {uri}")
    return uri


def launch(s3_data_uri, s3_code_uri):
    sm = boto3.client("sagemaker", region_name=REGION)

    print(f"[launch] job={JOB_NAME}  instance={INSTANCE_TYPE}")

    sm.create_training_job(
        TrainingJobName=JOB_NAME,
        AlgorithmSpecification={
            "TrainingImage":     IMAGE_URI,
            "TrainingInputMode": "File",
        },
        RoleArn=ROLE,
        InputDataConfig=[{
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType":             "S3Prefix",
                    "S3Uri":                  s3_data_uri,
                    "S3DataDistributionType": "FullyReplicated",
                },
            },
            "CompressionType": "None",
        }],
        OutputDataConfig={
            "S3OutputPath": f"s3://{S3_BUCKET}/{S3_PREFIX}/output/",
        },
        ResourceConfig={
            "InstanceType":   INSTANCE_TYPE,
            "InstanceCount":  1,
            "VolumeSizeInGB": 60,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 36 * 3600},
        # Script-mode: DLC reads these two HPs to find the entry script
        HyperParameters={
            "sagemaker_program":            "train_tiled.py",
            "sagemaker_submit_directory":   s3_code_uri,
            "model":                        "yolo11s.pt",
            "epochs":                       "100",
            "imgsz":                        "640",
        },
    )

    print(f"\n✓ Job submitted: {JOB_NAME}")
    print(f"  Console: https://console.aws.amazon.com/sagemaker/home"
          f"?region={REGION}#/jobs/{JOB_NAME}")
    print(f"  Output:  s3://{S3_BUCKET}/{S3_PREFIX}/output/"
          f"{JOB_NAME}/output/model.tar.gz")
    print(f"\n  Track:  aws sagemaker describe-training-job "
          f"--training-job-name {JOB_NAME} "
          f"--query '{{Status:TrainingJobStatus}}'")
    print(f"\n  Download when done:")
    print(f"    aws s3 cp s3://{S3_BUCKET}/{S3_PREFIX}/output/"
          f"{JOB_NAME}/output/model.tar.gz .")
    print(f"    tar xzf model.tar.gz   # extracts detector_best.pt")
    print(f"\n  Instance auto-terminates on Completed/Failed.")
    return JOB_NAME


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-upload", action="store_true")
    a = p.parse_args()

    s3_data = upload_dataset(skip=a.skip_upload)
    s3_code = upload_sourcecode()
    launch(s3_data, s3_code)


if __name__ == "__main__":
    main()
