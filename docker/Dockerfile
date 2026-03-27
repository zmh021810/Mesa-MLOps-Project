# 1. 使用 Lambda Python 3.9 镜像作为基础
FROM public.ecr.aws/lambda/python:3.9

# 2. 设置工作目录（Lambda 默认路径）
WORKDIR ${LAMBDA_TASK_ROOT}

# 3. 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 拷贝整个 src 目录（包含你新建的 5 个文件夹和 __init__.py）
# 这一步执行后，容器内会有 /var/task/src/predictor/predict.py
COPY src/ ./src/

# 5. 确保 Python 能识别 src 为顶级包
ENV PYTHONPATH="${PYTHONPATH}:${LAMBDA_TASK_ROOT}"

# 6. 🌟 关键修改：指向新的路径
# 格式：包名.子包名.文件名.函数名
CMD [ "src.predictor.predict.handler" ]