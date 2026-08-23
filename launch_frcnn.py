"""SageMaker launcher for the torchvision Faster R-CNN two-stage detector
(GRSL revision cross-architecture experiment). Tiled dataset already in S3."""
import os, tarfile, tempfile, time, boto3
REGION="us-east-1"; ACCOUNT="YOUR_ACCOUNT_ID"
ROLE=f"arn:aws:iam::{ACCOUNT}:role/service-role/YOUR_SAGEMAKER_ROLE"
BUCKET=f"sagemaker-us-east-1-{ACCOUNT}"; PREFIX="maritime-sgtile"
IMAGE=f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker"
S3_DATA=f"s3://{BUCKET}/{PREFIX}/tiles/"
job=f"frcnn-sds-{int(time.time())}"
s3=boto3.client("s3",region_name=REGION)
with tempfile.NamedTemporaryFile(suffix=".tar.gz",delete=False) as tmp: tar=tmp.name
with tarfile.open(tar,"w:gz") as t:
    t.add("train/train_frcnn.py",arcname="train_frcnn.py")
key=f"{PREFIX}/code/{job}/sourcedir.tar.gz"; s3.upload_file(tar,BUCKET,key); os.unlink(tar)
code=f"s3://{BUCKET}/{key}"
sm=boto3.client("sagemaker",region_name=REGION)
sm.create_training_job(
    TrainingJobName=job,
    AlgorithmSpecification={"TrainingImage":IMAGE,"TrainingInputMode":"File"},
    RoleArn=ROLE,
    InputDataConfig=[{"ChannelName":"training","DataSource":{"S3DataSource":{
        "S3DataType":"S3Prefix","S3Uri":S3_DATA,"S3DataDistributionType":"FullyReplicated"}},
        "CompressionType":"None"}],
    OutputDataConfig={"S3OutputPath":f"s3://{BUCKET}/{PREFIX}/output/"},
    ResourceConfig={"InstanceType":"ml.g5.2xlarge","InstanceCount":1,"VolumeSizeInGB":80},
    StoppingCondition={"MaxRuntimeInSeconds":36*3600},
    HyperParameters={"sagemaker_program":"train_frcnn.py","sagemaker_submit_directory":code,
        "epochs":"12","batch-size":"4","lr":"0.005"})
print("SUBMITTED",job)
