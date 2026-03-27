import argparse
import boto3
import json
import time

def get_sagemaker_role(iam_client, role_name):
    """辅助函数：通过角色名获取完整的 ARN"""
    response = iam_client.get_role(RoleName=role_name)
    return response['Role']['Arn']

def run_train(sm_client, iam_client, bucket, image_uri, role_name, data_key):
    """命令 SageMaker 启动一个计算实例进行训练"""
    job_name = f"mesa-pro-train-{int(time.time())}"
    role_arn = get_sagemaker_role(iam_client, role_name)
    
    print(f"🚀 [SageMaker] 正在创建训练任务: {job_name}")
    
    sm_client.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            'TrainingImage': image_uri,
            'TrainingInputMode': 'File',
            'ContainerEntrypoint': ["python", "-m", "src.train"],
            'ContainerArguments': [
                "--bucket", bucket,
                "--data_key", data_key
            ]
        },
        RoleArn=role_arn,
        InputDataConfig=[{
            'ChannelName': 'train',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': f"s3://{bucket}/train/", # 默认数据源
                    'S3DataDistributionType': 'FullyReplicated',
                }
            }
        }],
        OutputDataConfig={'S3OutputPath': f"s3://{bucket}/model-artifacts/"},
        ResourceConfig={
            'InstanceType': 'ml.m5.large',
            'InstanceCount': 1,
            'VolumeSizeInGB': 5,
        },
        StoppingCondition={'MaxRuntimeInSeconds': 3600}
    )
    print(f"✅ 任务已提交！请前往控制台查看日志。")

def run_test(ld_client, function_name, test_data_key):
    """
    命令 Lambda 立即执行一次预测
    test_data_key: S3 中的测试文件路径，例如 'test/test_data.csv'
    """
    if not test_data_key:
        print("❌ 错误：执行 test 动作必须提供数据路径。请使用 --path 参数。")
        return

    print(f"🧠 [Lambda] 正在触发推理请求: {function_name}")
    print(f"📄 测试数据源: {test_data_key}")
    
    payload = {"data_key": test_data_key}
    
    # 调用 Lambda
    response = ld_client.invoke(
        FunctionName=function_name,
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )
    
    # 解析并打印结果
    result = json.loads(response['Payload'].read().decode())
    print(f"🎉 预测结果反馈:\n{json.dumps(result, indent=2)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesa MLOps Pro 统一控制台")
    parser.add_argument('action', choices=['train', 'test'], help="执行动作")
    
    # 基础资源参数
    parser.add_argument('--bucket', required=True, help="S3 桶名")
    parser.add_argument('--image', required=True, help="ECR 镜像 URI")
    parser.add_argument('--role', required=True, help="IAM Role 名称")
    
    # 目标参数
    parser.add_argument('--target', required=True, help="Lambda函数名 (针对test) 或 训练任务名称 (针对train)")
    
    # 灵活的数据路径参数
    parser.add_argument('--path', help="S3 中的数据文件路径 (例如: test/my_data.csv 或 train/input.csv)")

    args = parser.parse_args()

    # 初始化 AWS 客户端
    session = boto3.Session()
    sm = session.client('sagemaker')
    ld = session.client('lambda')
    iam = session.client('iam')

    if args.action == 'train':
        # 如果 train 没传 path，可以默认用 'train/' 目录
        data_path = args.path if args.path else "train/"
        run_train(sm, iam, args.bucket, args.image, args.role, data_path)
        
    elif args.action == 'test':
        # 如果 test 没传 path，报错或提醒
        if not args.path:
            print("⚠️ 警告：未提供 --path，将尝试默认路径 'test/test_data.csv'")
            test_path = "test/test_data.csv"
        else:
            test_path = args.path
            
        run_test(ld, args.target, test_path)