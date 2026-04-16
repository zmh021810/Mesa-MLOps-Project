import csv
from aws_cdk import (
    Stack,
    RemovalPolicy,
    Duration,
    CfnOutput,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_iam as iam,
    aws_ecr_assets as ecr_assets,
)


class MesaMlopsProStack(Stack):
    def __init__(self, scope, construct_id, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # --- 1. Load Client List ---
        # Reads client names from a CSV file to drive multi-tenant resource creation
        clients = []
        try:
            with open("clients.csv", mode="r", encoding="utf-8") as f:
                clients = [row[0].strip() for row in csv.reader(f) if row]
        except FileNotFoundError:
            # Fallback for local testing or CI/CD environments
            print("Warning: clients.csv not found.")

        # --- 2. Shared Docker Image Asset ---
        # Unified image for both SageMaker Training and Batch Transformation
        ml_image_asset = ecr_assets.DockerImageAsset(
            self, "MesaProImage", directory=".", file="docker/Dockerfile"
        )

        # --- 3. Shared SageMaker Execution Role ---
        # Service role used by SageMaker instances to access AWS resources
        sagemaker_role = iam.Role(
            self,
            "MesaSageMakerRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                )
            ],
        )

        # --- 4. Multi-Tenant Storage (Isolated S3 Buckets) ---
        # Create a dedicated physical bucket for each external client to ensure data isolation
        client_buckets = {}
        for client in clients:
            bucket = s3.Bucket(
                self,
                f"Bucket-{client}",
                bucket_name=f"mesa-mlops-{client.lower()}-storage",
                versioned=True,
                removal_policy=RemovalPolicy.DESTROY,  # Objects will be deleted on stack deletion
                auto_delete_objects=True,
            )
            client_buckets[client] = bucket

        # --- 5. Dispatcher Lambda (The Orchestrator) ---
        # Lightweight function to trigger SageMaker jobs based on incoming requests
        dispatcher_lambda = _lambda.Function(
            self,
            "MesaDispatcher",
            runtime=_lambda.Runtime.PYTHON_3_9,
            handler="dispatcher.handler",
            code=_lambda.Code.from_asset("src/dispatcher"),
            timeout=Duration.seconds(60),
            environment={
                "ML_IMAGE_URI": ml_image_asset.image_uri,
                "SAGEMAKER_ROLE_ARN": sagemaker_role.role_arn,
            },
        )

        # --- 6. Security & Permission Mapping ---
        # Grant specific access rights to the shared roles for each tenant's bucket
        for client, bucket in client_buckets.items():
            # Allow SageMaker to read training data and write model/inference results
            bucket.grant_read_write(sagemaker_role)
            # Allow Dispatcher to inspect bucket contents for automated routing
            bucket.grant_read(dispatcher_lambda)

        # --- 7. Dispatcher IAM Policy Expansion ---
        # Grants Dispatcher the ability to manage SageMaker jobs and pass the execution role
        dispatcher_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:CreateTransformJob",
                    "sagemaker:CreateModel",
                    "iam:PassRole",
                ],
                resources=[
                    "*"
                ],  # Scoping can be narrowed down to specific job ARNs if needed
            )
        )

        # --- 8. Infrastructure Outputs ---
        # Export key resource details for reference or external integration
        CfnOutput(self, "ML_IMAGE_URI", value=ml_image_asset.image_uri)
        CfnOutput(self, "SageMakerRoleArn", value=sagemaker_role.role_arn)
