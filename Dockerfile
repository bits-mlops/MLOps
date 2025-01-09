FROM python:3.8-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Azure Environment Variables
ENV AZURE_TENANT_ID=""
ENV AZURE_CLIENT_ID=""
ENV AZURE_CLIENT_SECRET=""
ENV AZURE_SUBSCRIPTION_ID=""
ENV AZURE_RESOURCE_GROUP=""
ENV AZURE_WORKSPACE_NAME=""

COPY requirements.txt .
COPY src/ src/
COPY tests/ tests/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python" ,"src/app.py"]