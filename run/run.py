import argparse
import boto3
import json
import time


def run_train(ld_client, dispatcher_name, client_id):
    """
    Triggers the Dispatcher Lambda to start a SageMaker Training Job.
    """
    print(f"🚀 [Dispatcher] Triggering Training for Client: {client_id}")

    payload = {"action": "train", "client_id": client_id}

    response = ld_client.invoke(
        FunctionName=dispatcher_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload),
    )

    result = json.loads(response["Payload"].read().decode())
    print(f"✅ Dispatcher Response:\n{json.dumps(result, indent=2)}")


def run_predict(ld_client, dispatcher_name, client_id):
    """
    Triggers the Dispatcher Lambda to launch a Scalable Batch Transform Job.
    The Dispatcher will automatically calculate the required InstanceCount.
    """
    print(f"🧠 [Dispatcher] Triggering Batch Prediction for Client: {client_id}")

    payload = {"action": "predict", "client_id": client_id}

    response = ld_client.invoke(
        FunctionName=dispatcher_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload),
    )

    result = json.loads(response["Payload"].read().decode())
    print(f"🎉 Batch Job Orchestrated:\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesa MLOps Pro Unified Console")

    # Action: train or predict
    parser.add_argument(
        "action", choices=["train", "predict"], help="Action to execute"
    )

    # Required: The name of the Dispatcher Lambda function
    parser.add_argument(
        "--dispatcher",
        required=True,
        help="Name of the Dispatcher Lambda function (e.g., MesaMlopsDispatcher)",
    )

    # Required: The Client ID to ensure multi-tenant isolation
    parser.add_argument(
        "--client", required=True, help="Client ID (e.g., ClientA, ClientB)"
    )

    args = parser.parse_args()

    # Initialize AWS Lambda Client
    # The console now only needs to talk to the Dispatcher
    session = boto3.Session()
    ld = session.client("lambda")

    if args.action == "train":
        run_train(ld, args.dispatcher, args.client)

    elif args.action == "predict":
        run_predict(ld, args.dispatcher, args.client)
