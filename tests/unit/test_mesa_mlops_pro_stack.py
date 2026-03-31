import aws_cdk as core
import aws_cdk.assertions as assertions

from mesa_mlops_pro.mesa_mlops_pro_stack import MesaMlopsProStack


# example tests. To run these tests, uncomment this file along with the example
# resource in mesa_mlops_pro/mesa_mlops_pro_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = MesaMlopsProStack(app, "mesa-mlops-pro")
    template = assertions.Template.from_stack(stack)


#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
