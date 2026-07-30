#!/usr/bin/env python3
"""
RunPod serverless handler for VoxCPM2 (openbmb/VoxCPM2, Apache-2.0).

Muc dich: engine giong cao cap cho tikvn.io-tts. Model 2.3B, tokenizer-free,
ho tro tieng Viet san. Chay GPU (fp16, ~5GB VRAM).

Mode suy ra tu input:
  clone — ref_audio (+ ref_text) provided -> voice cloning
  auto  — khong co ref -> giong mac dinh cua model

Input:
{
  "input": {
    "text":          str,             # required
    "ref_audio_url":    str,          # optional (clone) OR
    "ref_audio_base64": str,
    "ref_text":         str,          # optional transcript cua ref
    "cfg_value":        float,        # default 2.0
    "inference_timesteps": int,       # default 10 (cao hon = min hon, cham hon)
    "normalize":        bool,         # default true (chuan hoa so/ky hieu noi bo)
    "output_format":    "mp3"|"wav",  # default "mp3"
    "r2": { "endpoint_url","access_key_id","secret_access_key","bucket_name" }
  }
}

Output: { "success": true, "mode", "audio_base64"|"audio_url", "r2_key",
          "duration_seconds", "processing_time_seconds" }
"""

import base64
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import runpod
import requests
import soundfile as sf

# Lazy-loaded, giu trong GPU giua cac request
_model = None
_sr: Optional[int] = None
_whisper = None

MODEL_ID = "openbmb/VoxCPM2"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def get_model():
    """Lazy-load VoxCPM2 tren GPU (fp16). load_denoiser=False: khoi phu thuoc model
    zipenhancer cua Alibaba (license rieng) + nhanh hon; optimize=False: tranh
    torch.compile treo o cold start."""
    global _model, _sr
    if _model is None:
        from voxcpm import VoxCPM
        log(f"Loading VoxCPM2 ({MODEL_ID}) on cuda...")
        _model = VoxCPM.from_pretrained(
            MODEL_ID, load_denoiser=False, optimize=False, device="cuda"
        )
        # Tim sample rate that
        for obj in (_model, getattr(_model, "tts_model", None), getattr(_model, "model", None)):
            if obj is None:
                continue
            for attr in ("sample_rate", "sr", "target_sample_rate"):
                v = getattr(obj, attr, None)
                if isinstance(v, (int, float)) and v > 1000:
                    _sr = int(v)
                    break
            if _sr:
                break
        _sr = _sr or 16000
        log(f"VoxCPM2 loaded (sample_rate={_sr})")
    return _model


def get_whisper():
    """Lazy-load faster-whisper de tu phien am clip mau (VoxCPM2 clone BAT BUOC prompt_text).
    Uu tien GPU; loi (cudnn mismatch...) thi fallback CPU (clip 15s CPU cung nhanh)."""
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        try:
            _whisper = WhisperModel("base", device="cuda", compute_type="float16")
            log("Whisper loaded on cuda")
        except Exception as e:  # noqa: BLE001
            log(f"Whisper cuda failed ({e}); fallback cpu")
            _whisper = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper


def transcribe_ref(path: str) -> str:
    """Tra ve transcript cua clip mau. Rong thi tra chuoi rong (noi goi tu xu ly)."""
    segs, _info = get_whisper().transcribe(path)
    return " ".join(s.text.strip() for s in segs).strip()


# --- I/O helpers ------------------------------------------------------------

def download_file(url: str, output_path: Path, timeout: int = 300) -> bool:
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        log(f"Download error: {e}")
        return False


def decode_base64_file(data: str, output_path: Path) -> bool:
    try:
        if "," in data:
            data = data.split(",", 1)[1]
        output_path.write_bytes(base64.b64decode(data))
        return True
    except Exception as e:
        log(f"Base64 decode error: {e}")
        return False


def encode_file_base64(file_path: Path) -> str:
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def get_audio_duration(audio_path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def wav_to_mp3(wav_path: Path, mp3_path: Path) -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame",
             "-b:a", "192k", str(mp3_path)],
            capture_output=True, timeout=120, check=True,
        )
        return mp3_path.exists()
    except Exception as e:
        log(f"WAV to MP3 error: {e}")
        return False


def upload_to_r2(file_path: Path, job_id: str, r2: dict, content_type: str):
    try:
        import boto3
        from botocore.config import Config
        client = boto3.client(
            "s3", endpoint_url=r2["endpoint_url"],
            aws_access_key_id=r2["access_key_id"],
            aws_secret_access_key=r2["secret_access_key"],
            config=Config(signature_version="s3v4"),
        )
        key = f"voxcpm2/results/{job_id}_{uuid.uuid4().hex[:8]}{file_path.suffix}"
        client.upload_file(str(file_path), r2["bucket_name"], key,
                           ExtraArgs={"ContentType": content_type})
        url = client.generate_presigned_url(
            "get_object", Params={"Bucket": r2["bucket_name"], "Key": key}, ExpiresIn=7200)
        return url, key
    except Exception as e:
        log(f"R2 upload error: {e}")
        return None, None


# --- Text chunking (VoxCPM cung bo chu khi input qua dai) --------------------

DEFAULT_CHUNK_CHARS = 240
CHUNK_GAP_SECONDS = 0.12


def _split_sentences(line: str) -> list:
    m = re.findall(r'[^.!?…]+[.!?…]+["\')\]]*|[^.!?…]+$', line)
    return [s.strip() for s in m if s.strip()] or ([line.strip()] if line.strip() else [])


def _hard_wrap(s: str, max_chars: int) -> list:
    out, cur = [], ""
    for tok in (re.findall(r'\S+', s) or [s]):
        if cur and len(cur) + 1 + len(tok) > max_chars:
            out.append(cur); cur = tok
        else:
            cur = (cur + " " + tok) if cur else tok
    if cur:
        out.append(cur)
    return out


def _split_long(sentence: str, max_chars: int) -> list:
    if len(sentence) <= max_chars:
        return [sentence]
    clauses = re.split(r'(?<=[,;:—-])\s+', sentence)
    merged, cur = [], ""
    for c in clauses:
        if cur and len(cur) + 1 + len(c) > max_chars:
            merged.append(cur); cur = c
        else:
            cur = (cur + " " + c) if cur else c
    if cur:
        merged.append(cur)
    out = []
    for mm in merged:
        out.extend([mm] if len(mm) <= max_chars else _hard_wrap(mm, max_chars))
    return out


def chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list:
    chunks, cur = [], ""
    for line in re.split(r'\r?\n+', text):
        line = line.strip()
        if not line:
            continue
        for sentence in _split_sentences(line):
            for piece in _split_long(sentence, max_chars):
                if cur and len(cur) + 1 + len(piece) > max_chars:
                    chunks.append(cur.strip()); cur = piece
                else:
                    cur = (cur + " " + piece) if cur else piece
    if cur.strip():
        chunks.append(cur.strip())
    return chunks or ([text.strip()] if text.strip() else [])


# --- Handler ----------------------------------------------------------------

def handler(job: dict) -> dict:
    job_id = job.get("id", "unknown")
    ji = job.get("input", {})
    t0 = time.time()

    text = ji.get("text")
    if not text:
        return {"error": "Missing required field: text"}

    output_format = ji.get("output_format", "mp3")
    r2 = ji.get("r2")
    cfg_value = float(ji.get("cfg_value", 2.0))
    inference_timesteps = int(ji.get("inference_timesteps", 10))
    normalize = bool(ji.get("normalize", True))

    work_dir = Path(tempfile.mkdtemp(prefix=f"voxcpm2_{job_id}_"))
    try:
        ref_audio_path = None
        ref_text = ji.get("ref_text")
        if ji.get("ref_audio_url"):
            ref_audio_path = work_dir / "ref.wav"
            if not download_file(ji["ref_audio_url"], ref_audio_path):
                return {"error": "Failed to download reference audio"}
        elif ji.get("ref_audio_base64"):
            ref_audio_path = work_dir / "ref.wav"
            if not decode_base64_file(ji["ref_audio_base64"], ref_audio_path):
                return {"error": "Failed to decode reference audio"}

        mode = "clone" if ref_audio_path is not None else "auto"
        model = get_model()

        # VoxCPM2 clone BAT BUOC prompt_text. Clone ma khach khong gui ref_text -> tu phien am
        # bang Whisper (khach chi can upload clip, khong phai go loi thoai).
        if ref_audio_path is not None and not (ref_text and ref_text.strip()):
            ref_text = transcribe_ref(str(ref_audio_path))
            log(f"Job {job_id}: auto-transcribed ref -> {len(ref_text)} chars")

        def gen_one(t: str) -> np.ndarray:
            kw = dict(text=t, cfg_value=cfg_value,
                      inference_timesteps=inference_timesteps, normalize=normalize)
            if ref_audio_path is not None:
                kw["prompt_wav_path"] = str(ref_audio_path)
                if ref_text:
                    kw["prompt_text"] = ref_text
            return np.asarray(model.generate(**kw), dtype=np.float32).reshape(-1)

        text_chunks = chunk_text(text)
        log(f"Job {job_id}: mode={mode}, {len(text_chunks)} chunk(s)")
        if len(text_chunks) <= 1:
            wav = gen_one(text_chunks[0] if text_chunks else text)
        else:
            gap = np.zeros(int(_sr * CHUNK_GAP_SECONDS), dtype=np.float32)
            parts = []
            for i, ch in enumerate(text_chunks):
                parts.append(gen_one(ch))
                if i < len(text_chunks) - 1:
                    parts.append(gap)
            wav = np.concatenate(parts)

        wav_path = work_dir / "out.wav"
        sf.write(str(wav_path), wav, _sr)

        if output_format == "mp3":
            out_path = work_dir / "out.mp3"
            if not wav_to_mp3(wav_path, out_path):
                return {"error": "Failed to convert WAV to MP3"}
            content_type = "audio/mpeg"
        else:
            out_path, content_type = wav_path, "audio/wav"

        elapsed = time.time() - t0
        result = {
            "success": True,
            "mode": mode,
            "processing_time_seconds": round(elapsed, 2),
            "duration_seconds": round(get_audio_duration(out_path), 2),
        }
        if r2:
            url, key = upload_to_r2(out_path, job_id, r2, content_type)
            if not url:
                return {"error": "Failed to upload to R2"}
            result["audio_url"] = url
            result["r2_key"] = key
        else:
            result["audio_base64"] = encode_file_base64(out_path)
        log(f"Job {job_id}: done in {elapsed:.1f}s")
        return result

    except Exception as e:
        import traceback
        log(f"Handler exception: {e}\n{traceback.format_exc()}")
        return {"error": f"Internal error: {e}"}
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    log("Starting RunPod VoxCPM2 handler...")
    try:
        import torch
        log("CUDA available" if torch.cuda.is_available() else "WARNING: CUDA not available!")
    except ImportError:
        pass
    runpod.serverless.start({"handler": handler})
