import os
import boto3
import json
import math
from datetime import datetime

# Initialize AWS Clients
sm_client = boto3.client("sagemaker")
s3_resource = boto3.resource("s3")
s3_client = boto3.client("s3")


def get_s3_folder_size(bucket_name, prefix):
    """
    Calculates the total size of objects under a specific S3 prefix in bytes.
    Used to dynamically estimate required compute power for prediction tasks.
    """
    bucket = s3_resource.Bucket(bucket_name)
    total_size = sum(obj.size for obj in bucket.objects.filter(Prefix=prefix))
    return total_size


def get_latest_model_key(bucket, prefix):
    """
    Automatically discovers the most recent model.tar.gz within a client's dedicated bucket.
    """
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in response:
        raise Exception(f"No model artifacts found in path: {prefix}")

    # Filter for model artifacts
    model_files = [
        obj for obj in response["Contents"] if obj["Key"].endswith("model.tar.gz")
    ]
    if not model_files:
        raise Exception("No model.tar.gz file found.")

    # Sort by last modified timestamp to get the freshest model
    latest_model = max(model_files, key=lambda x: x["LastModified"])
    return latest_model["Key"]


def handler(event, context):
    try:
        # 1. Request Parsing
        # Handle both direct Lambda invocation and API Gateway proxy integration
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        else:
            body = event.get("body", {})

        action = body.get("action")  # Expected: 'train' or 'predict'
        client_id = body.get("client_id")

        if not client_id:
            return {"statusCode": 400, "body": "Missing client_id"}

        # 2. Environment & Resource Setup
        # Use ML_IMAGE_URI (Unified image for all ML tasks)
        client_bucket = f"mesa-mlops-{client_id.lower()}-storage"
        image_uri = os.environ["ML_IMAGE_URI"]
        role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # --- ACTION A: Cloud-based Training ---
        if action == "train":
            job_name = f"train-{client_id.lower()}-{timestamp}"

            print(f"🚀 Launching SageMaker Training for Client: {client_id}")
            sm_client.create_training_job(
                TrainingJobName=job_name,
                AlgorithmSpecification={
                    "TrainingImage": image_uri,
                    "TrainingInputMode": "File",
                    # 🌟 ENTRYPOINT OVERRIDE: Directly trigger the training module
                    "ContainerEntrypoint": ["python", "-m", "src.train.train"],
                },
                RoleArn=role_arn,
                InputDataConfig=[
                    {
                        "ChannelName": "train",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": f"s3://{client_bucket}/train/",
                            }
                        },
                    }
                ],
                OutputDataConfig={
                    "S3OutputPath": f"s3://{client_bucket}/model-artifacts/"
                },
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

        # --- ACTION B: Asynchronous Batch Prediction ---
        elif action == "predict":
            input_prefix = "batch_input/"

            # Dynamic Instance Scaling Logic
            total_bytes = get_s3_folder_size(client_bucket, input_prefix)
            total_gb = total_bytes / (1024**3)
            calculated_instances = math.ceil(total_gb / 1.0)
            instance_count = max(1, min(calculated_instances, 5))

            print(
                f"📊 Client: {client_id} | Data Size: {total_gb:.2f} GB | Instances: {instance_count}"
            )

            # 1. Prepare Model Artifacts
            latest_key = get_latest_model_key(client_bucket, "model-artifacts/")
            model_s3_path = f"s3://{client_bucket}/{latest_key}"
            model_name = f"model-{client_id.lower()}-{timestamp}"
            transform_job_name = f"predict-{client_id.lower()}-{timestamp}"

            # 2. Register SageMaker Model Object
            print(f"📦 Registering Model Object: {model_name}")
            sm_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    "Image": image_uri,
                    "ModelDataUrl": model_s3_path,
                    # 🌟 ENTRYPOINT OVERRIDE: Directly trigger the prediction module
                    # No 'MODE' environment variable needed anymore
                    "ContainerEntrypoint": ["python", "-m", "src.predict.predict"],
                },
                ExecutionRoleArn=role_arn,
            )

            # 3. Launch Batch Transform Job
            print(f"🚀 Launching Parallel Batch Transform: {transform_job_name}")
            sm_client.create_transform_job(
                TransformJobName=transform_job_name,
                ModelName=model_name,
                MaxPayloadInMB=6,
                BatchStrategy="MultiRecord",
                TransformInput={
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{client_bucket}/{input_prefix}",
                        }
                    },
                    "ContentType": "text/csv",
                    "SplitType": "Line",
                },
                TransformOutput={
                    "S3OutputPath": f"s3://{client_bucket}/batch_output/",
                    "AssembleWith": "Line",
                },
                TransformResources={
                    "InstanceType": "ml.m5.large",
                    "InstanceCount": instance_count,
                },
            )

            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "msg": "Batch transform job successfully orchestrated",
                        "job_name": transform_job_name,
                        "instances": instance_count,
                    }
                ),
            }

        return {"statusCode": 400, "body": json.dumps({"error": "Unsupported action"})}

    except Exception as e:
        print(f"❌ Dispatcher Error: {str(e)}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
