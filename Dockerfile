FROM python:3.15-slim

WORKDIR /code
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# CPU-only torch first. The default Linux wheel bundles CUDA and is several GB.
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Mock engine by default. The adapter is NOT baked in; it is mounted or downloaded later.
ENV ENGINE=mock
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]