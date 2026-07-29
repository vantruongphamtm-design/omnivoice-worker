# Gwen-TTS RunPod Serverless Worker (Qwen3-TTS-0.6B fine-tune tieng Viet, MIT).
# Voice CLONE tieng Viet goc, thay VieNeu. GPU/bf16. Model bake san (cold start thap).
#
# cu128 + runtime: GIONG worker VieNeu, 2 ly do:
#   1) cu128 (torch 2.8) cho GPU RunPod AMPERE_24 (co Blackwell sm_120).
#   2) `runtime` (khong `devel`): dung attn "sdpa" -> KHONG can flash-attn/nvcc (build nhanh).
# LUU Y bf16 HONG tren T4/RTX20 -> dung GPU Ampere+ (AMPERE_24 OK).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# ffmpeg cho WAV->MP3 + atempo + ffprobe; git de clone package qwen_tts.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*

# Deps runtime. KHONG cai flash-attn (engine dung attn_implementation="sdpa").
RUN pip install --no-cache-dir "transformers>=4.51" soundfile librosa einops accelerate runpod boto3 requests

# Cai package qwen_tts tu repo gwen-tts. --no-deps: KHONG keo torch cu128 cua base, KHONG keo flash-attn.
# TODO(build): neu import qwen_tts loi thieu dep -> them dep do vao dong pip tren.
RUN git clone --depth 1 https://github.com/ggroup-ai-lab/gwen-tts.git /app/gwen-tts \
 && pip install --no-cache-dir --no-deps -e /app/gwen-tts \
 && pip check || echo "[warn] pip check co canh bao (xem lai neu load loi)"
ENV PYTHONPATH=/app/gwen-tts:${PYTHONPATH}

WORKDIR /app
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# === BAKE MODEL CHINH VAO IMAGE (giam cold start) ===
# KHONG set HF_HUB_OFFLINE: cho phep tai codec/tokenizer phu (neu co) luc chay lan dau.
# TODO(build): neu smoke test bao thieu codec repo -> bake them repo do o day.
RUN python -c "from huggingface_hub import snapshot_download; print('Downloading Gwen-TTS 0.6B...'); snapshot_download('g-group-ai-lab/gwen-tts-0.6B'); print('Model baked.')"

COPY gwen_engine.py /app/gwen_engine.py
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
