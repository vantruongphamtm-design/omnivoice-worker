# OmniVoice RunPod Serverless Worker (model baked in)
# Omnilingual zero-shot TTS / voice cloning (k2-fsa/OmniVoice), fp32 inference.
# torch>=2.5 base is REQUIRED: omnivoice needs transformers>=5.3.0, which imports
# DTensor from the public torch.distributed.tensor (added in torch 2.5). The old
# torch 2.4 base broke with "cannot import name 'DTensor'".
#
# Build (from repo root):
#   docker buildx build --platform linux/amd64 -t ghcr.io/<your-namespace>/video-toolkit-omnivoice:latest --push docker/runpod-omnivoice/
#
# Cold start: ~30-60s (no model download needed, weights baked in)
#
# Test locally:
#   docker run --gpus all -p 8000:8000 ghcr.io/<your-namespace>/video-toolkit-omnivoice:latest

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Install ffmpeg for WAV -> MP3 conversion
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# omnivoice pulls in its own torch/torchaudio/transformers pins — let pip resolve them
RUN pip install --no-cache-dir \
    omnivoice \
    runpod \
    boto3 \
    requests

WORKDIR /app

# Set HF cache location
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# === BAKE MODEL INTO IMAGE ===
RUN python -c "\
from huggingface_hub import snapshot_download; \
import os; \
os.makedirs('/root/.cache/huggingface', exist_ok=True); \
print('Downloading OmniVoice model...'); \
snapshot_download('k2-fsa/OmniVoice', cache_dir='/root/.cache/huggingface'); \
print('OmniVoice model downloaded successfully'); \
"

# Copy handler
COPY handler.py /app/handler.py

# Start handler
CMD ["python", "-u", "/app/handler.py"]
