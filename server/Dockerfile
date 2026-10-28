FROM python:3.12-slim
WORKDIR /app
# compiler only needed to build pySTEPS' C extensions
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# server plus the backtest scripts (verify_*.py, compare_pysteps.py)
COPY *.py *.json ./
EXPOSE 8771
CMD ["python", "-u", "rainy_server.py"]
