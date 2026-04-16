FROM python:3.9-slim

# 关键：确保 Python 路径正确，这样 Dispatcher 调用命令时才不会报错
ENV PYTHONPATH="/opt/ml/code:${PYTHONPATH}"
WORKDIR /opt/ml/code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 只需要把代码拷进去，不需要 entrypoint.sh，不需要 CMD
COPY src/ ./src/