# VoxCPM2 RunPod Serverless Worker (model baked + dep-slim -> image nho hon, cold start ngan hon)
# openbmb/VoxCPM2 (Apache-2.0), 2.3B tokenizer-free TTS. GPU fp16 (~5GB VRAM).
#
# Base cu128 (torch 2.8): BAT BUOC vi RunPod tier 24GB co the xep RTX PRO 6000 Blackwell MIG (sm_120);
# base cu124 cu se "no kernel image" -> worker unhealthy (bai hoc omnivoice 19/07).
#
# GHI CHU: model VAN bake vao image (khong dung network volume) vi volume RunPod chu yeu o Secure
# Cloud (GPU dat hon RTX 4090/3090 cong dong dang dung) + khoa vung. Neu sau nay xac minh duoc vung
# co GPU re + volume, doi HF_HOME=/runpod-volume/hf va bo 2 buoc bake duoi.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*

# voxcpm keo torch/transformers/... theo pin cua no. faster-whisper: tu phien am clip mau cho clone.
RUN pip install --no-cache-dir voxcpm faster-whisper runpod boto3 requests

# SLIM: go cac goi CHI dung cho UI demo / plotting / training-data (KHONG dinh inference clone).
# Da xac minh cuc bo: `from voxcpm import VoxCPM` KHONG nap cac goi nay -> go an toan, image nho hon ~1-2GB.
RUN pip uninstall -y gradio gradio-client hf-gradio safehttpx spaces matplotlib contourpy datasets 2>/dev/null || true

WORKDIR /app
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# === BAKE MODEL INTO IMAGE === (khoi tai 5GB moi lan cold start)
RUN python -c "\
from huggingface_hub import snapshot_download; \
import os; \
os.makedirs('/root/.cache/huggingface', exist_ok=True); \
print('Downloading VoxCPM2...', flush=True); \
snapshot_download('openbmb/VoxCPM2', cache_dir='/root/.cache/huggingface'); \
print('VoxCPM2 downloaded', flush=True); \
"

# Bake whisper base model (tranh tai luc cold start)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8'); print('whisper base baked', flush=True)"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
