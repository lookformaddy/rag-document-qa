FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ragqa ./ragqa
COPY examples ./examples
EXPOSE 8000
CMD ["python", "-m", "ragqa.cli", "serve"]
