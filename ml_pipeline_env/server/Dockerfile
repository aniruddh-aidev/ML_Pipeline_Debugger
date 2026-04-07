FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# Copy environment package
COPY . /app/ml_pipeline_env/

# Install dependencies
COPY ml_pipeline_env/server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "ml_pipeline_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
