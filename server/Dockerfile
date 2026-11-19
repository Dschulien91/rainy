FROM python:3.12-slim
WORKDIR /app
# compiler only needed to build pySTEPS' C extensions
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# server plus the backtest scripts (verify_*.py, compare_pysteps.py)
# + terrain for the Europe shadow "oro" (dem_eu4km.npz, build_dem.py)
COPY *.py *.json *.npz ./
# fewer malloc arenas: with many threads and big numpy temporaries glibc kept freeing memory per thread arena
# and the process grew until the Docker VM killed it (exit 137 after hours)
ENV MALLOC_ARENA_MAX=2
EXPOSE 8771
CMD ["python", "-u", "rainy_server.py"]
