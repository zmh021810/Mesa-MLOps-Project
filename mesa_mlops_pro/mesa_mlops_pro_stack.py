from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_iam as iam,
    aws_ecr_assets as ecr_assets,
    CfnOutput,
    Duration,
    RemovalPolicy,
)
from constructs import Construct

class MesaMlopsProStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # 1. 创建 S3 存储桶 (保持不变)
        bucket = s3.Bucket(
            self,
            "MesaProStorage",
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # 2. 定义镜像 Asset (用于 SageMaker 训练和 Predictor)
        ml_image_asset = ecr_assets.DockerImageAsset(
            self,
            "MesaProImage",
            directory=".",
            file="docker/Dockerfile",
        )

        # 3. 创建预测 Lambda (Predictor)
        # 修改点：handler 还是叫 handler，但代码逻辑下午我们会更新
        predictor_lambda = _lambda.DockerImageFunction(
            self,
            "MesaPredictor",
            code=_lambda.DockerImageCode.from_image_asset(
                directory=".",
                file="docker/Dockerfile",
                cmd=["src.predictor.predict.handler"]
            ),
            memory_size=1024,
            timeout=Duration.seconds(150),
            environment={
                "BUCKET_NAME": bucket.bucket_name,
            },
        )

        # 4. 创建 SageMaker IAM 角色
        sagemaker_role = iam.Role(
            self,
            "MesaSageMakerRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
            ],
        )

        # 🌟 5. 新增：创建指挥官 Lambda (Dispatcher)
        # 这是一个轻量级 Lambda，直接用 Python 脚本，不需要 Docker 镜像
        dispatcher_lambda = _lambda.Function(
            self,
            "MesaDispatcher",
            runtime=_lambda.Runtime.PYTHON_3_9,
            handler="dispatcher.handler",
            code=_lambda.Code.from_asset("src/dispatcher"), # 记得新建这个文件夹
            timeout=Duration.seconds(30),
            environment={
                "BUCKET_NAME": bucket.bucket_name,
                "TRAIN_IMAGE_URI": ml_image_asset.image_uri,
                "SAGEMAKER_ROLE_ARN": sagemaker_role.role_arn,
                "PREDICT_FUNCTION_NAME": predictor_lambda.function_name,
            }
        )

        # 🌟 6. 权限交叉配置
        # A. 允许 Predictor 读写 S3
        bucket.grant_read_write(predictor_lambda)
        # B. 允许 SageMaker 读写 S3
        bucket.grant_read_write(sagemaker_role)
        # C. 允许 Dispatcher 列出 S3 文件 (用于“自动寻路”) 并调用训练/预测
        bucket.grant_read(dispatcher_lambda)
        predictor_lambda.grant_invoke(dispatcher_lambda)
        
        # D. 赋予 Dispatcher 启动 SageMaker 的权限
        dispatcher_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["sagemaker:CreateTrainingJob", "iam:PassRole"],
            resources=["*"]
        ))

        # 🌟 7. 开放 Function URL (让你可以从本地直接 curl 指挥)
        dispatcher_url = dispatcher_lambda.add_function_url(
            auth_type=_lambda.FunctionUrlAuthType.NONE # 生产环境建议后续加 IAM 校验
        )

        # 8. 关键输出
        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "DispatcherUrl", value=dispatcher_url.url) # 以后你只需记住这个 URL
        CfnOutput(self, "LambdaFunctionName", value=predictor_lambda.function_name)
        CfnOutput(self, "SageMakerRoleArn", value=sagemaker_role.role_arn)
        CfnOutput(self, "ImageUri", value=ml_image_asset.image_uri)