FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY quandry-2024.0.0.dev0-py3-none-any.whl .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py"]
