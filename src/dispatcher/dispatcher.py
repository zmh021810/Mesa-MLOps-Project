import boto3
import json
import os
import time

# 初始化 AWS 客户端
sm_client = boto3.client("sagemaker")
lambda_client = boto3.client("lambda")
s3_client = boto3.client("s3")


def get_latest_model_key(bucket, prefix):
    """
    自动寻路逻辑：在指定的 S3 路径下寻找最新的 model.tar.gz
    """
    print(f"🔍 正在寻找最新模型，路径: s3://{bucket}/{prefix}")

    # 列出该前缀下的所有对象
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if "Contents" not in response:
        raise Exception(f"S3 路径下没有找到任何模型文件: {prefix}")

    # 过滤出所有的 model.tar.gz 文件
    model_files = [
        obj for obj in response["Contents"] if obj["Key"].endswith("model.tar.gz")
    ]

    if not model_files:
        raise Exception("未找到任何 model.tar.gz 文件")

    # 按修改时间排序，取最新的一个
    latest_model = max(model_files, key=lambda x: x["LastModified"])
    print(
        f"✅ 找到最新模型: {latest_model['Key']} (更新时间: {latest_model['LastModified']})"
    )
    return latest_model["Key"]


def handler(event, context):
    try:
        # 1. 解析请求指令
        body = json.loads(event.get("body", "{}"))
        action = body.get("action")  # 'train' 或 'test'

        # 从环境变量获取资源配置（这些由 CDK 注入）
        bucket = os.environ["BUCKET_NAME"]
        image_uri = os.environ["TRAIN_IMAGE_URI"]
        role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
        predict_fn_name = os.environ["PREDICT_FUNCTION_NAME"]

        # --- 动作 A: 启动云端训练 ---
        if action == "train":
            job_name = f"mesa-train-{int(time.time())}"
            data_target = body.get("target", "train/train_data.csv")  # 默认训练集路径

            print(f"🚀 启动 SageMaker 训练任务: {job_name}")
            sm_client.create_training_job(
                TrainingJobName=job_name,
                AlgorithmSpecification={
                    "TrainingImage": image_uri,
                    "TrainingInputMode": "File",
                    "ContainerEntrypoint": ["python", "-m", "src.train.train"],
                    "ContainerArguments": ["--target", data_target],
                },
                RoleArn=role_arn,
                InputDataConfig=[
                    {
                        "ChannelName": "train",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": f"s3://{bucket}/train/",
                                "S3DataDistributionType": "FullyReplicated",
                            }
                        },
                    }
                ],
                OutputDataConfig={"S3OutputPath": f"s3://{bucket}/model-artifacts/"},
                ResourceConfig={
                    "InstanceType": "ml.m5.large",
                    "InstanceCount": 1,
                    "VolumeSizeInGB": 5,
                },
                StoppingCondition={"MaxRuntimeInSeconds": 3600},
            )
            return {
                "statusCode": 200,
                "body": json.dumps(
                    {"msg": "Training job started", "job_name": job_name}
                ),
            }

        # --- 动作 B: 触发自动预测 ---
        elif action == "test":
            # 1. 自动定位最新模型
            latest_key = get_latest_model_key(bucket, "model-artifacts/")

            # 2. 准备发给 Predictor 的“包裹”
            payload = {
                "bucket": bucket,
                "model_key": latest_key,
                "data_key": body.get(
                    "data_key", "test/test_data.csv"
                ),  # 默认测试集路径
            }

            print(f"📞 正在呼叫预测 Lambda: {predict_fn_name}")
            resp = lambda_client.invoke(
                FunctionName=predict_fn_name,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload),
            )

            # 3. 解析并返回预测结果
            result = json.loads(resp["Payload"].read().decode())
            return {"statusCode": 200, "body": json.dumps(result)}

        return {"statusCode": 400, "body": "Invalid action"}

    except Exception as e:
        print(f"❌ Dispatcher 崩溃: {str(e)}")
        return {"statusCode": 500, "body": str(e)}
