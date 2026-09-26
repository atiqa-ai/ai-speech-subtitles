# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- builder ---
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the CPU-only torch wheel first and explicitly. The default PyPI
# torch wheel is the multi-gigabyte CUDA build, which would bloat the image
# for no benefit: this pipeline runs Whisper on the CPU.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.14.0+cpu"

COPY requirements.txt ./
RUN pip install -r requirements.txt

# ---------------------------------------------------------------- runtime ---
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="AI Speech Subtitles" \
      org.opencontainers.image.description="Lecture video to Urdu subtitle pipeline"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY main.py ./
COPY src/ ./src/

# Tests ship with the image so the container's dependency set can be verified
# directly in CI, rather than trusting that the host build is representative.
COPY tests/ ./tests/
COPY requirements.txt requirements-dev.txt ./

# No system ffmpeg is installed on purpose: src/ffmpeg_setup.py resolves the
# binary that ships inside imageio-ffmpeg, which is what lets this project run
# without a separate ffmpeg install.

RUN mkdir -p /app/output
VOLUME ["/app/output"]

# A one-shot CLI has no long-running process to health check. CI instead asserts
# that `main.py --help` works, which exercises the full import graph.

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
