FROM python:3.12-slim

WORKDIR /app

# Poetry 설치
RUN pip install --upgrade pip && pip install poetry

# 소스 복사 및 의존성 설치
COPY pyproject.toml .
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi
COPY . .

EXPOSE 50053

CMD ["python", "grpc_server.py"] 
