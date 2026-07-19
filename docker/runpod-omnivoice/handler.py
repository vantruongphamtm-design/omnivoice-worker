#!/usr/bin/env python3
"""
RunPod serverless handler for OmniVoice (k2-fsa/OmniVoice).

Single unified model; mode is inferred from which inputs are provided
(mirrors the official `omnivoice-infer` CLI):

  clone        — ref_audio (+ ref_text) provided -> voice cloning
  voice_design — instruct provided, no ref_audio  -> generate a new voice
                 from a natural-language style brief
  auto         — neither provided                 -> model's default voice

Input format:
{
    "input": {
        "text":          str,             # required
        "language":      str,             # optional, e.g. "English"

        # clone
        "ref_audio_url":    str,          # OR
        "ref_audio_base64": str,
        "ref_text":         str,          # optional transcript (auto-transcribed if omitted)

        # voice_design
        "instruct": str,                  # style/character brief

        # generation params (all optional, defaults match the CLI)
        "num_step": int,                  # default 32
        "guidance_scale": float,          # default 2.0
        "speed": float,                   # default 1.0
        "duration": float,                # fixed output duration in seconds
        "t_shift": float,                 # default 0.1
        "denoise": bool,                  # default true
        "output_format": "mp3" | "wav",   # default "mp3"

        # R2 (optional; else base64 returned)
        "r2": {
            "endpoint_url":      str,
            "access_key_id":     str,
            "secret_access_key": str,
            "bucket_name":       str
        }
    }
}

Output format:
{
    "success": true,
    "mode": str,
    "processing_time_seconds": float,
    "audio_url":        str,   # if R2 configured
    "r2_key":           str,
    "audio_base64":     str,   # if no R2
    "duration_seconds": float
}
"""

import base64
import re
import shutil
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

# Lazy-loaded, kept in GPU memory between requests
_model = None

MODEL_ID = "k2-fsa/OmniVoice"


def log(message: str) -> None:
    """Log message to stderr (visible in RunPod logs)."""
    print(message, file=sys.stderr, flush=True)


def get_model():
    """Lazy-load OmniVoice in fp16 ("16bit")."""
    global _model
    if _model is None:
        import torch
        from omnivoice.models.omnivoice import OmniVoice

        log(f"Loading OmniVoice ({MODEL_ID}) in fp32...")
        _model = OmniVoice.from_pretrained(
            MODEL_ID, device_map="cuda:0", dtype=torch.float32
        )
        log("OmniVoice model loaded")
    return _model


# ─── I/O helpers (same pattern as the qwen3-tts worker) ─────

def download_file(url: str, output_path: Path, timeout: int = 300) -> bool:
    try:
        log(f"Downloading from {url[:80]}...")
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        log(f"  Downloaded: {output_path.name} ({output_path.stat().st_size // 1024}KB)")
        return True
    except Exception as e:
        log(f"Download error: {e}")
        return False


def decode_base64_file(data: str, output_path: Path) -> bool:
    try:
        if "," in data:
            data = data.split(",", 1)[1]
        decoded = base64.b64decode(data)
        output_path.write_bytes(decoded)
        log(f"Decoded base64 to {output_path.name} ({len(decoded) // 1024}KB)")
        return True
    except Exception as e:
        log(f"Base64 decode error: {e}")
        return False


def encode_file_base64(file_path: Path) -> str:
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def get_audio_duration(audio_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception as e:
        log(f"Error getting audio duration: {e}")
        return 0.0


def wav_to_mp3(wav_path: Path, mp3_path: Path) -> bool:
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(wav_path),
                "-codec:a", "libmp3lame",
                "-b:a", "192k",
                str(mp3_path),
            ],
            capture_output=True, timeout=120, check=True,
        )
        return mp3_path.exists()
    except Exception as e:
        log(f"WAV to MP3 conversion error: {e}")
        return False


def upload_to_r2(file_path: Path, job_id: str, r2_config: dict, content_type: str) -> tuple[Optional[str], Optional[str]]:
    """Upload audio to Cloudflare R2 and return (presigned_url, object_key)."""
    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=r2_config["endpoint_url"],
            aws_access_key_id=r2_config["access_key_id"],
            aws_secret_access_key=r2_config["secret_access_key"],
            config=Config(signature_version="s3v4"),
        )
        ext = file_path.suffix
        object_key = f"omnivoice/results/{job_id}_{uuid.uuid4().hex[:8]}{ext}"
        client.upload_file(
            str(file_path),
            r2_config["bucket_name"],
            object_key,
            ExtraArgs={"ContentType": content_type},
        )
        presigned_url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": r2_config["bucket_name"], "Key": object_key},
            ExpiresIn=7200,
        )
        return presigned_url, object_key
    except Exception as e:
        log(f"R2 upload error: {e}")
        return None, None


# ─── Text chunking ──────────────────────────────────────────
# OmniVoice (như mọi TTS clone zero-shot) BỎ CHỮ / nhảy đoạn khi input quá dài. Cắt theo
# RANH GIỚI CÂU rồi ghép các mảng audio lại -> 1 file hoàn chỉnh, đọc đủ, nghe liền mạch.

DEFAULT_CHUNK_CHARS = 240
CHUNK_GAP_SECONDS = 0.12  # khoảng lặng nhỏ chèn giữa các khúc khi ghép


def _split_sentences(line: str) -> list:
    matches = re.findall(r'[^.!?…]+[.!?…]+["\')\]]*|[^.!?…]+$', line)
    if not matches:
        return [line.strip()] if line.strip() else []
    return [s.strip() for s in matches if s.strip()]


def _hard_wrap(s: str, max_chars: int) -> list:
    tokens = re.findall(r'\[[^\]]*\]|\S+', s) or [s]
    out, cur = [], ""
    for tok in tokens:
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
    for m in merged:
        out.extend([m] if len(m) <= max_chars else _hard_wrap(m, max_chars))
    return out


def chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> list:
    """Cắt văn bản thành các khúc <= max_chars theo ranh giới câu, giữ nguyên thẻ [..]."""
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
    if chunks:
        return chunks
    return [text.strip()] if text.strip() else []


# ─── Handler ────────────────────────────────────────────────

def handler(job: dict) -> dict:
    """Main RunPod handler for OmniVoice."""
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})
    start_time = time.time()

    log(f"Job {job_id}: Starting OmniVoice")

    text = job_input.get("text")
    if not text:
        return {"error": "Missing required field: text"}

    language = job_input.get("language")
    output_format = job_input.get("output_format", "mp3")
    r2_config = job_input.get("r2")
    instruct = job_input.get("instruct")

    work_dir = Path(tempfile.mkdtemp(prefix=f"omnivoice_{job_id}_"))
    log(f"Working directory: {work_dir}")

    try:
        ref_audio_path = None
        ref_text = job_input.get("ref_text")

        if job_input.get("ref_audio_url"):
            ref_audio_path = work_dir / "ref_audio.wav"
            if not download_file(job_input["ref_audio_url"], ref_audio_path):
                return {"error": "Failed to download reference audio"}
        elif job_input.get("ref_audio_base64"):
            ref_audio_path = work_dir / "ref_audio.wav"
            if not decode_base64_file(job_input["ref_audio_base64"], ref_audio_path):
                return {"error": "Failed to decode reference audio"}

        if ref_audio_path is not None:
            mode = "clone"
        elif instruct:
            mode = "voice_design"
        else:
            mode = "auto"

        model = get_model()

        gen_kwargs = dict(
            text=text,
            language=language,
            ref_audio=str(ref_audio_path) if ref_audio_path else None,
            ref_text=ref_text,
            instruct=instruct,
            duration=job_input.get("duration"),
            num_step=int(job_input.get("num_step", 32)),
            guidance_scale=float(job_input.get("guidance_scale", 2.0)),
            speed=float(job_input.get("speed", 1.0)),
            t_shift=float(job_input.get("t_shift", 0.1)),
            denoise=bool(job_input.get("denoise", True)),
        )

        log(f"Job {job_id}: mode={mode}, generating...")

        # Cắt văn bản dài theo câu rồi ghép -> model không bỏ chữ. CHỈ cắt khi KHÔNG ép
        # duration (duration dùng để khớp khung phụ đề SRT; áp per-khúc sẽ sai thời lượng).
        force_duration = job_input.get("duration")
        text_chunks = [text] if force_duration else chunk_text(text)

        if len(text_chunks) <= 1:
            wav = np.asarray(model.generate(**gen_kwargs)[0], dtype=np.float32)
        else:
            log(f"Job {job_id}: text dai -> cat {len(text_chunks)} khuc roi ghep")
            gap = np.zeros(int(model.sampling_rate * CHUNK_GAP_SECONDS), dtype=np.float32)
            parts = []
            for i, chunk in enumerate(text_chunks):
                seg_kwargs = dict(gen_kwargs, text=chunk)
                parts.append(np.asarray(model.generate(**seg_kwargs)[0], dtype=np.float32))
                if i < len(text_chunks) - 1:
                    parts.append(gap)
            wav = np.concatenate(parts)

        wav_path = work_dir / "output.wav"
        sf.write(str(wav_path), wav, model.sampling_rate)

        if output_format == "mp3":
            out_path = work_dir / "output.mp3"
            if not wav_to_mp3(wav_path, out_path):
                return {"error": "Failed to convert WAV to MP3"}
            content_type = "audio/mpeg"
        else:
            out_path = wav_path
            content_type = "audio/wav"

        duration = get_audio_duration(out_path)
        elapsed = time.time() - start_time
        log(f"Job {job_id}: generated in {elapsed:.1f}s")

        result = {
            "success": True,
            "mode": mode,
            "processing_time_seconds": round(elapsed, 2),
            "duration_seconds": round(duration, 2),
        }

        if r2_config:
            url, r2_key = upload_to_r2(out_path, job_id, r2_config, content_type)
            if not url:
                return {"error": "Failed to upload to R2"}
            result["audio_url"] = url
            result["r2_key"] = r2_key
        else:
            result["audio_base64"] = encode_file_base64(out_path)

        return result

    except Exception as e:
        import traceback
        log(f"Handler exception: {e}")
        log(traceback.format_exc())
        return {"error": f"Internal error: {str(e)}"}
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
            log("Cleaned up working directory")
        except Exception:
            pass


# RunPod serverless entry point
if __name__ == "__main__":
    log("Starting RunPod OmniVoice handler...")

    try:
        import torch
        if torch.cuda.is_available():
            log(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            log("WARNING: CUDA not available!")
    except ImportError:
        log("Warning: torch not imported for CUDA check")

    runpod.serverless.start({"handler": handler})
