FROM apache/airflow:3.3.2-python3.12
# Fixar apache-airflow impede o pip de atualizá-lo; as versões de ML batem com o uv.lock.
RUN pip install --no-cache-dir \
    "apache-airflow==3.3.2" \
    "scikit-learn==1.9.1" \
    "skl2onnx==1.20.0" \
    "onnx==1.23.0" \
    "onnxruntime==1.30.0"
ENV PYTHONPATH=/opt/airflow/src
