# VieNeu-TTS v3 Turbo RunPod Serverless Worker (model baked in)
# Voice CLONE tieng Viet goc (Apache-2.0), thay OmniVoice (weights CC-BY-NC).
# GPU/bf16 inference. Cold start thap nho bake san model (khong tai HF luc chay).
#
# Build (tu repo root):
#   docker buildx build --platform linux/amd64 \
#     -t ghcr.io/vantruongphamtm-design/vieneu-worker:latest --push docker/runpod-vieneu/
#
# cu128 + runtime: GIONG worker OmniVoice, cung 2 ly do da do:
#   1) cu128 (torch 2.8) BAT BUOC — RunPod AMPERE_24 co ca RTX PRO 6000 Blackwell MIG
#      (sm_120); base cu124/torch2.6 khong co kernel -> worker "unhealthy" ngau nhien.
#   2) `runtime` (khong `devel`): handler khong torch.compile/cpp_extension -> khong can nvcc.
# LUU Y bf16 HONG tren T4/RTX20-series -> dung GPU Ampere+ (AMPERE_24 OK).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# ffmpeg cho WAV->MP3, atempo va ffprobe do duration
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Cai theo thu tu VieNeu doc GPU: base da co torch 2.8.0 cu128; them torchaudio khop,
# transformers pin, roi vieneu (clone chay tren PyTorch engine). Cac dep audio cua engine
# v3 Turbo (librosa/einops) cai truoc de tranh ImportError luc load. pip vieneu (KHONG [gpu])
# khong keo torch (deps torch nam trong extra [gpu]) nen KHONG de len torch cu128 cua base.
# TODO(build): neu load loi thieu dep -> them dep do (xac minh tren build 3.2.3 that, khong phai 3.0.12).
RUN pip install --no-cache-dir torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir "transformers==4.57.6" librosa einops huggingface_hub \
 && pip install --no-cache-dir "vieneu==3.2.3" runpod boto3 requests soundfile \
 && pip check || echo "[warn] pip check co canh bao (xem lai neu load loi)"

WORKDIR /app
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# === BAKE MODEL + CODEC VAO IMAGE (khong tai HF luc chay -> cold start thap) ===
# TODO(build): pin revision="<sha>" cho ca 2 repo khi da chot ban v3 Turbo (dang preview).
RUN python -c "\
from huggingface_hub import snapshot_download; \
print('Downloading VieNeu-TTS v3 Turbo...'); \
snapshot_download('pnnbao-ump/VieNeu-TTS-v3-Turbo'); \
print('Downloading MOSS audio codec...'); \
snapshot_download('OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano'); \
print('Models baked.'); \
"

# Sau khi bake: chay OFFLINE luc runtime -> khong tai HF nua (cold start thap, khong phu thuoc
# mang, khong bi preview repo doi lam cache miss). Bake o tren van ONLINE (truoc dong ENV nay).
# Neu smoke test tren RunPod bao thieu file khi offline -> go 2 dong nay (cache van hit binh thuong).
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Copy handler + engine wrapper
COPY vieneu_engine.py /app/vieneu_engine.py
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
