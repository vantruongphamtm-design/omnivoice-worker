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

# Base: cu128 + runtime. Hai lý do, đều đã đo:
#
# 1) cu128 (torch 2.8) LÀ BẮT BUỘC. RunPod tier AMPERE_24 không chỉ có RTX 3090 —
#    nó còn xếp cả RTX PRO 6000 Blackwell MIG 1g.24gb (sm_120). Base cu124/torch2.6
#    không có kernel cho sm_120 nên worker rơi vào "no kernel image is available for
#    execution on the device" và chết thành `unhealthy`, ngẫu nhiên tuỳ máy được cấp.
#    Sự cố 2026-07-19 (worker unhealthy trên PRO 6000 MIG 24GB) là biểu hiện của lỗi này.
#
# 2) `runtime` thay cho `devel`. Bản devel gánh cả CUDA toolkit (nvcc, headers, static
#    libs) nặng nhiều GB. handler.py chỉ import numpy/runpod/requests/soundfile, không
#    có torch.compile hay cpp_extension, nên KHÔNG cần trình biên dịch lúc chạy.
#
# omnivoice 0.2.1 yêu cầu torch>=2.4 (không chặn trên) nên torch 2.8 thoả mãn,
# pip sẽ không cài đè một bản torch thứ hai.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

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
