FROM registry.access.redhat.com/ubi9/python-311:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /opt/app

# Install dependencies
COPY app/requirements.txt /opt/app/requirements.txt
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

# Copy application
COPY app /opt/app

# Expose and run with uvicorn
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
