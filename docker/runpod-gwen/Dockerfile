# Gwen-TTS RunPod Serverless Worker (Qwen3-TTS-0.6B fine-tune tieng Viet, MIT).
# Voice CLONE tieng Viet goc, thay VieNeu. GPU/bf16. Model bake san (cold start thap).
#
# cu128 + runtime: GIONG worker VieNeu, 2 ly do:
#   1) cu128 (torch 2.8) cho GPU RunPod AMPERE_24 (co Blackwell sm_120).
#   2) `runtime` (khong `devel`): engine dung attn "sdpa" -> KHONG can flash-attn/nvcc (build nhanh).
#      qwen-tts 0.1.1 KHONG doi flash-attn (deps: transformers/accelerate/librosa/torchaudio/onnxruntime).
# LUU Y bf16 HONG tren T4/RTX20 -> dung GPU Ampere+ (AMPERE_24 OK).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# ffmpeg cho WAV->MP3 + atempo + ffprobe; sox cho python-sox (dep cua qwen-tts).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg sox && rm -rf /var/lib/apt/lists/*

# torchaudio khop torch 2.8 cu128 (tranh PyPI keo ban le lech CUDA). Roi qwen-tts (package cung cap
# `from qwen_tts import Qwen3TTSModel` + generate_voice_clone) keo transformers==4.57.3, accelerate,
# librosa, onnxruntime, einops... torch da co san (base) nen pip KHONG cai lai torch.
# huggingface_hub<1.0: transformers 4.57.x doi (loi da gap o worker VieNeu: found hub 1.x).
RUN pip install --no-cache-dir torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir "qwen-tts==0.1.1" faster-whisper runpod boto3 requests \
 && pip install --no-cache-dir --no-deps --force-reinstall "huggingface_hub<1.0" \
 && pip check || echo "[warn] pip check co canh bao (xem lai neu load loi)"

WORKDIR /app
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# === BAKE MODEL CHINH VAO IMAGE (giam cold start) ===
# KHONG set HF_HUB_OFFLINE: cho phep tai codec/tokenizer phu (neu qwen-tts can) luc chay lan dau.
RUN python -c "from huggingface_hub import snapshot_download; print('Downloading Gwen-TTS 0.6B...'); snapshot_download('g-group-ai-lab/gwen-tts-0.6B'); print('Model baked.')"

# Bake whisper 'small' (ASR fallback khi user clone khong nhap transcript). Tai model files
# luc build (device cpu int8 chi de download); runtime chay lai tren GPU float16.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8'); print('Whisper small baked.')"

COPY gwen_engine.py /app/gwen_engine.py
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
