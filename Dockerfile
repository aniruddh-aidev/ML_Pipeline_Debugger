FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# Copy everything into the root of /app
COPY . .

# Use the actual path revealed by your 'ls' command
RUN pip install --no-cache-dir -r server/requirements.txt

# Ensure Python sees both folders as packages
ENV PYTHONPATH=/app

EXPOSE 7860

# Point to the app.py inside the server folder
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]