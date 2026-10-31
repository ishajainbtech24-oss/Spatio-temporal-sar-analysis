# SAR Flood Intelligence - production image.
# Works on Hugging Face Spaces (Docker SDK), Render, Railway, Google Cloud Run, or any Docker host.
#
#   docker build -t sar-flood .                      # full: includes TensorFlow + legacy classifier
#   docker build -t sar-flood --build-arg WITH_TF=0 .  # lite: no TensorFlow, ~4x smaller, fits 512 MB hosts
#   docker run -p 7860:7860 sar-flood                # open http://localhost:7860
FROM python:3.12-slim

ARG WITH_TF=1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PORT=7860 \
    RESULT_TTL_HOURS=24 \
    TRUST_PROXY=1 \
    SAR_FLOOD_LEGACY=${WITH_TF}

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /app

# Dependencies first (cached between code changes).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[deploy]" \
 && if [ "$WITH_TF" = "1" ]; then pip install "tensorflow-cpu>=2.16,<3"; fi

COPY . .
RUN mkdir -p /app/outputs && chown -R user:user /app
USER user

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)"

# One worker keeps memory low (models load once); threads handle concurrent visitors.
CMD exec gunicorn --bind "0.0.0.0:${PORT}" --workers 1 --threads 4 --timeout 180 \
    --access-logfile - "sar_flood.web.app:create_app()"
