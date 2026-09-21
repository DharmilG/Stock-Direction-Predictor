FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md COMMANDS.txt .env.example ./
COPY config ./config
COPY src ./src
COPY start_run.py start_run.bat start_run.sh ./
RUN pip install --no-cache-dir -e .
EXPOSE 8000
CMD ["uvicorn", "stockml.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
