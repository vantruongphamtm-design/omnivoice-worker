# VoxCPM2 RunPod Serverless Worker (model baked in)
# openbmb/VoxCPM2 (Apache-2.0), 2.3B tokenizer-free TTS, ho tro tieng Viet. GPU fp16 (~5GB VRAM).
#
# Base cu128 (torch 2.8): BAT BUOC vi RunPod tier 24GB co the xep RTX PRO 6000 Blackwell
# MIG (sm_120); base cu124 cu se "no kernel image" -> worker unhealthy (bai hoc omnivoice 19/07).
# runtime (khong devel): handler khong compile gi luc chay nen khong can nvcc.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*

# voxcpm keo torch/transformers/... theo pin cua no. Tren Linux, PyPI torch la ban CUDA
# (khong phai CPU nhu khi cai bang --index-url cpu o may local), nen GPU van chay.
# faster-whisper: tu phien am clip mau cho clone (VoxCPM2 bat buoc prompt_text)
RUN pip install --no-cache-dir voxcpm faster-whisper runpod boto3 requests

WORKDIR /app
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# === BAKE MODEL INTO IMAGE === (khoi phai tai 5GB moi lan cold start)
RUN python -c "\
from huggingface_hub import snapshot_download; \
import os; \
os.makedirs('/root/.cache/huggingface', exist_ok=True); \
print('Downloading VoxCPM2...', flush=True); \
snapshot_download('openbmb/VoxCPM2', cache_dir='/root/.cache/huggingface'); \
print('VoxCPM2 downloaded', flush=True); \
"

# Bake whisper base model (tranh tai luc cold start; nap CPU chi de cache CT2 model)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8'); print('whisper base baked', flush=True)"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
