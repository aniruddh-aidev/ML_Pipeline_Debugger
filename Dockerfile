FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r server/requirements.txt

# This makes absolute imports like 'from server.app' work
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]