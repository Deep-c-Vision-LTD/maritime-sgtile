"""
Launch tiled-detector training on Amazon SageMaker (PyTorch DLC for Ultralytics).

Prereqs:
  * tiled dataset staged in S3 (data/slice_dataset.py output): data.yaml + images/ + labels/
  * SageMaker execution role

Usage:
  python launch_sagemaker.py \
      --s3-data   s3://my-bucket/sgtile/sds_tiles \
      --s3-output s3://my-bucket/sgtile/output \
      --role      arn:aws:iam::<acct>:role/<SageMakerExecutionRole> \
      --model yolo11s.pt --epochs 100
"""

import argparse
import os

import sagemaker
from sagemaker.pytorch import PyTorch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s3-data", required=True)
    p.add_argument("--s3-output", required=True)
    p.add_argument("--role", default=None)
    p.add_argument("--instance-type", default="ml.g5.2xlarge")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    role = args.role or sagemaker.get_execution_role()
    estimator = PyTorch(
        entry_point="train_tiled.py",
        source_dir=os.path.dirname(os.path.abspath(__file__)),
        role=role,
        instance_count=1,
        instance_type=args.instance_type,
        framework_version="2.3",
        py_version="py311",
        hyperparameters={
            "model": args.model, "epochs": args.epochs,
            "batch-size": args.batch_size, "imgsz": args.imgsz,
        },
        output_path=args.s3_output,
        base_job_name="sgtile-detector",
    )
    estimator.fit({"training": args.s3_data})
    print("Model artifact:", estimator.model_data)


if __name__ == "__main__":
    main()
