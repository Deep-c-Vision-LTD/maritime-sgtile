"""
Fine-tune the maritime target-aware saliency prior on SageMaker (TF GPU).

Pipeline:
  1. Pre-resize train images to the training resolution (small → cheap upload).
  2. Sync resized images + target maps to S3.
  3. Launch a TensorFlow DLC training job running saliency/finetune.py.

The ml.g5.xlarge auto-terminates on completion. Encoder is frozen, ~650 MB
data, short run (<1 h, <$2).

Usage:
  AWS_PROFILE=xsoyel-sm python launch_saliency_sagemaker.py            # full
  AWS_PROFILE=xsoyel-sm python launch_saliency_sagemaker.py --skip-prep
"""

import argparse
import glob
import os
import sys
import tarfile
import tempfile
import time

import boto3

REGION   = "us-east-1"
ACCOUNT  = "YOUR_ACCOUNT_ID"
ROLE     = (f"arn:aws:iam::{ACCOUNT}:role/service-role/"
            "YOUR_SAGEMAKER_ROLE")
BUCKET   = f"sagemaker-us-east-1-{ACCOUNT}"
PREFIX   = "maritime-sgtile/saliency-ft"
INSTANCE = "ml.g5.xlarge"
# TensorFlow 2.16 training DLC (ships Keras 3). Short alias tag verified working
# by the prior successful xsoyelnet-finetune job on this same account/region.
IMAGE    = (f"763104351884.dkr.ecr.{REGION}.amazonaws.com/"
            "tensorflow-training:2.16-gpu-py310")
H, W     = 384, 640
JOB      = f"saliency-sar-{int(time.time())}"

SRC_IMG  = "data/seadronessee/images/train"
SRC_MAP  = "data/seadronessee/target_maps/train"
RES_IMG  = "data/work/sal_ft/images/train"   # local resized copies


def prep_images(skip=False):
    if skip:
        print("[prep] skipping resize")
        return
    import cv2
    os.makedirs(RES_IMG, exist_ok=True)
    maps = sorted(glob.glob(os.path.join(SRC_MAP, "*.png")))
    print(f"[prep] resizing {len(maps)} train images → {W}x{H} …")
    done = 0
    for mp in maps:
        stem = os.path.splitext(os.path.basename(mp))[0]
        src = os.path.join(SRC_IMG, stem + ".jpg")
        dst = os.path.join(RES_IMG, stem + ".jpg")
        if os.path.exists(dst) or not os.path.exists(src):
            continue
        img = cv2.imread(src)
        if img is None:
            continue
        cv2.imwrite(dst, cv2.resize(img, (W, H)))
        done += 1
        if done % 1000 == 0:
            print(f"[prep] {done} resized")
    print(f"[prep] resized {done} images → {RES_IMG}")


def sync_s3(skip=False):
    import subprocess
    img_uri = f"s3://{BUCKET}/{PREFIX}/images/train/"
    map_uri = f"s3://{BUCKET}/{PREFIX}/target_maps/train/"
    if skip:
        print("[s3] skipping upload")
        return f"s3://{BUCKET}/{PREFIX}/"
    for local, uri in [(RES_IMG, img_uri), (SRC_MAP, map_uri)]:
        print(f"[s3] sync {local} → {uri}")
        r = subprocess.run(["aws", "s3", "sync", local, uri, "--no-progress"])
        if r.returncode != 0:
            sys.exit(f"[s3] sync failed for {local}")
    return f"s3://{BUCKET}/{PREFIX}/"


def upload_code():
    """Tar finetune.py + model.py + losses.py (flat) + requirements.txt."""
    s3 = boto3.client("s3", region_name=REGION)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as t:
        code_tar = t.name
    with tarfile.open(code_tar, "w:gz") as tar:
        tar.add("saliency/finetune.py", arcname="finetune.py")
        tar.add("saliency/model.py",    arcname="model.py")
        tar.add("saliency/losses.py",   arcname="losses.py")
        # requirements: ensure opencv is present in the TF container
        req = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        req.write("opencv-python-headless\n")
        req.close()
        tar.add(req.name, arcname="requirements.txt")
    key = f"{PREFIX}/code/{JOB}/sourcedir.tar.gz"
    s3.upload_file(code_tar, BUCKET, key)
    os.unlink(code_tar)
    uri = f"s3://{BUCKET}/{key}"
    print(f"[code] → {uri}")
    return uri


def launch(data_uri, code_uri, epochs, batch):
    sm = boto3.client("sagemaker", region_name=REGION)
    print(f"[launch] {JOB}  instance={INSTANCE}")
    sm.create_training_job(
        TrainingJobName=JOB,
        AlgorithmSpecification={"TrainingImage": IMAGE, "TrainingInputMode": "File"},
        RoleArn=ROLE,
        InputDataConfig=[{
            "ChannelName": "training",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": data_uri,
                "S3DataDistributionType": "FullyReplicated"}},
            "CompressionType": "None",
        }],
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/{PREFIX}/output/"},
        ResourceConfig={"InstanceType": INSTANCE, "InstanceCount": 1,
                        "VolumeSizeInGB": 30},
        StoppingCondition={"MaxRuntimeInSeconds": 6 * 3600},
        HyperParameters={
            "sagemaker_program": "finetune.py",
            "sagemaker_submit_directory": code_uri,
            "epochs": str(epochs),
            "batch-size": str(batch),
            "infer-h": str(H), "infer-w": str(W),
            "freeze-encoder": "True",
        },
    )
    print(f"\n✓ Job submitted: {JOB}")
    print(f"  Console: https://console.aws.amazon.com/sagemaker/home"
          f"?region={REGION}#/jobs/{JOB}")
    print(f"  Output:  s3://{BUCKET}/{PREFIX}/output/{JOB}/output/model.tar.gz")
    print(f"  Track:   AWS_PROFILE=xsoyel-sm aws sagemaker describe-training-job "
          f"--training-job-name {JOB} --query TrainingJobStatus --output text")
    return JOB


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-prep", action="store_true", help="skip resize")
    p.add_argument("--skip-upload", action="store_true", help="skip S3 sync")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()

    prep_images(skip=a.skip_prep)
    data_uri = sync_s3(skip=a.skip_upload)
    code_uri = upload_code()
    launch(data_uri, code_uri, a.epochs, a.batch_size)


if __name__ == "__main__":
    main()
