# VoxCPM2 RunPod Serverless Worker — model tren NETWORK VOLUME (khong bake -> image nho, cold start nhanh)
# openbmb/VoxCPM2 (Apache-2.0), 2.3B tokenizer-free TTS. GPU fp16 (~5GB VRAM).
#
# Base cu128 (torch 2.8): BAT BUOC vi RunPod tier 24GB co the xep RTX PRO 6000 Blackwell MIG (sm_120);
# base cu124 cu se "no kernel image" -> worker unhealthy (bai hoc omnivoice 19/07).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*

# voxcpm keo torch/transformers/... theo pin cua no. faster-whisper: tu phien am clip mau cho clone.
RUN pip install --no-cache-dir voxcpm faster-whisper runpod boto3 requests

# SLIM: go cac goi CHI dung cho UI demo / plotting / training-data (KHONG dinh inference clone).
# Da xac minh cuc bo: `from voxcpm import VoxCPM` KHONG nap cac goi nay -> go an toan, image nho hon.
RUN pip uninstall -y gradio gradio-client hf-gradio safehttpx spaces matplotlib contourpy datasets 2>/dev/null || true

WORKDIR /app
# Model KHONG bake vao image. HF_HOME tro toi NETWORK VOLUME (mount o /runpod-volume). Cold start
# DAU TIEN tai model (~5GB) xuong volume (cham 1 lan); sau do moi worker doc tu volume, khoi keo lai.
ENV HF_HOME=/runpod-volume/hf
ENV PYTHONUNBUFFERED=1

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
