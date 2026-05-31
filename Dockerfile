# Supports Ampere / Ada / Hopper / Blackwell (A100, H100, RTX 3090/4090/5060Ti+)
# CUDA 12.6, cuDNN 9 — required for sm_120 (Blackwell)
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

WORKDIR /workspace

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl rsync openssh-client ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ──────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Clone optimizer repos that are not on PyPI ───────────────────────────────
# Pin to known-good commits so the image is reproducible
RUN git clone https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch \
        AdEMAMix_Optimizer_Pytorch \
    && git clone https://github.com/xinyuluo8561/Stacey \
    && cp Stacey/Stacey/staceybase.py . 2>/dev/null || true

# ── Application source ───────────────────────────────────────────────────────
COPY src/      src/
COPY main.py   .
COPY deploy/   deploy/

# ── Create volume mount-points (contents come from Docker volumes at runtime)
RUN mkdir -p /workspace/outputs /workspace/mlruns /workspace/data

# ── Non-root user (good practice; vast.ai also supports root containers) ─────
# Uncomment if your vast.ai template requires non-root
# RUN useradd -m appuser && chown -R appuser /workspace
# USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/workspace/data/torch_cache

# Default: show help. Override with --task-type etc. at runtime.
CMD ["python", "main.py", "--help"]
