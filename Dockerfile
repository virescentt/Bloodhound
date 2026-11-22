# Bloodhound — obraz CPU-only (bez GPU): interfejs Streamlit + CLI (python -m slad …).
#   docker compose up --build app        → http://localhost:8501   (więcej przykładów: docker-compose.yml)
#   docker build -t bloodhound . && docker run --rm -p 127.0.0.1:8501:8501 -v "$PWD/workspace:/app/workspace" bloodhound
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/workspace:/app/workspace" bloodhound python -m slad demo
# Dane NASK i wyniki NIE trafiają do obrazu (.dockerignore) — montujemy je jako wolumeny.
FROM python:3.13-slim

# 1 = także CLIP (open_clip + torch CPU, requirements-extra.txt; obraz większy o ≈1 GB)
ARG WITH_CLIP=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

# Biblioteki systemowe: glib (opencv-python-headless) i OpenMP (onnxruntime).
# Debian 13 (trixie) nazywa pakiet glib „libglib2.0-0t64”, starsze wydania „libglib2.0-0”.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && (apt-get install -y --no-install-recommends libglib2.0-0t64 \
        || apt-get install -y --no-install-recommends libglib2.0-0) \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-extra.txt ./
RUN pip install -r requirements.txt \
    && if [ "$WITH_CLIP" = "1" ]; then pip install -r requirements-extra.txt; fi

COPY . .
# Modele twarzy (YuNet + SFace, ≈40 MB) pobierane przy budowaniu; bez sieci obraz działa bez twarzy
RUN python -m slad models \
    || echo "UWAGA: modele twarzy nie zostały pobrane — rozpoznawanie twarzy wyłączone (python -m slad models)"
RUN mkdir -p data workspace reference/faces

EXPOSE 8501
CMD ["python", "-m", "slad", "ui", "--host", "0.0.0.0", "--port", "8501", "--no-browser"]
