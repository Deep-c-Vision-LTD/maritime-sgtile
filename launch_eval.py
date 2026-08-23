"""Launch a SageMaker GPU eval (baseline + S1) for one detector backend."""
import argparse, os, tarfile, tempfile, time, boto3
REGION="us-east-1"; ACCOUNT="YOUR_ACCOUNT_ID"
ROLE=f"arn:aws:iam::{ACCOUNT}:role/service-role/YOUR_SAGEMAKER_ROLE"
BUCKET=f"sagemaker-us-east-1-{ACCOUNT}"; PREFIX="maritime-sgtile"
IMAGE=f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker"
VD=f"s3://{BUCKET}/{PREFIX}/valdata"
p=argparse.ArgumentParser()
p.add_argument("--backend",required=True); p.add_argument("--weights-s3",required=True)
p.add_argument("--keep-thr",default="0.05"); p.add_argument("--conf",default="0.25"); p.add_argument("--alt",default="1")
a=p.parse_args()
job=f"eval-{a.backend}-{int(time.time())}"
s3=boto3.client("s3",region_name=REGION)
with tempfile.NamedTemporaryFile(suffix=".tar.gz",delete=False) as tmp: tar=tmp.name
with tarfile.open(tar,"w:gz") as t:
    t.add("sm_eval.py",arcname="sm_eval.py")
    t.add("tiling"); t.add("eval"); t.add("data/convert_seadronessee.py",arcname="data/convert_seadronessee.py")
key=f"{PREFIX}/code/{job}/sourcedir.tar.gz"; s3.upload_file(tar,BUCKET,key); os.unlink(tar)
code=f"s3://{BUCKET}/{key}"
def ch(uri): return {"S3DataSource":{"S3DataType":"S3Prefix","S3Uri":uri,"S3DataDistributionType":"FullyReplicated"}}
sm=boto3.client("sagemaker",region_name=REGION)
sm.create_training_job(
    TrainingJobName=job,
    AlgorithmSpecification={"TrainingImage":IMAGE,"TrainingInputMode":"File"},
    RoleArn=ROLE,
    InputDataConfig=[
        {"ChannelName":"images","DataSource":ch(f"{VD}/images/"),"CompressionType":"None"},
        {"ChannelName":"saliency","DataSource":ch(f"{VD}/saliency_sar/"),"CompressionType":"None"},
        {"ChannelName":"ann","DataSource":ch(f"{VD}/instances_val.json"),"CompressionType":"None"},
        {"ChannelName":"weights","DataSource":ch(a.weights_s3),"CompressionType":"None"}],
    OutputDataConfig={"S3OutputPath":f"s3://{BUCKET}/{PREFIX}/evalout/"},
    ResourceConfig={"InstanceType":"ml.g5.xlarge","InstanceCount":1,"VolumeSizeInGB":40},
    StoppingCondition={"MaxRuntimeInSeconds":6*3600},
    HyperParameters={"sagemaker_program":"sm_eval.py","sagemaker_submit_directory":code,
        "backend":a.backend,"keep_thr":a.keep_thr,"conf":a.conf,"alt":a.alt})
print("SUBMITTED",job)
