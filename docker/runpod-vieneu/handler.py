#!/usr/bin/env python3
"""RunPod serverless handler for VieNeu-TTS v3 Turbo (voice cloning).

DROP-IN thay OmniVoice: GIU NGUYEN contract input/output cua handler OmniVoice cu de
web (tts.ts) va video (tts.mjs) khong phai sua. Chi thay MODEL: Kokoro/F5-style OmniVoice
-> VieNeu v3 Turbo (Apache-2.0, tieng Viet goc, clone khong can ref_text).

Input (job["input"]):
{
    "text": str,                       # BAT BUOC
    "ref_audio_base64" | "ref_audio_url": str,   # clip mau de clone (clone khi co)
    "output_format": "mp3" | "wav",    # mac dinh "mp3"
    "duration": float,                 # ep do dai (giay) cho khop cue SRT (atempo)
    "speed": float,                    # >1 nhanh hon (atempo)
    "denoise": bool,                   # lam sach clip mau truoc khi enroll (mac dinh true)
    "r2": {endpoint_url, access_key_id, secret_access_key, bucket_name},  # up R2 thay base64
    # BO QUA (nhan nhung khong dung): ref_text, instruct, language,
    #   num_step, guidance_scale, t_shift, temperature
}
Output:
{
    "success": true, "mode": "clone"|"auto",
    "processing_time_seconds": float, "duration_seconds": float,
    "audio_url": str, "r2_key": str,   # neu co r2
    "audio_base64": str,               # neu khong co r2
}
Loi: {"error": "..."}
"""
import base64
import os
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

from vieneu_engine import VieNeuCloneEngine, match_loudness

# ── Engine singleton (nap 1 lan, giu am giua cac request) ──────────────────
_engine = None
R2_KEY_PREFIX = os.environ.get("R2_KEY_PREFIX", "omnivoice/results")  # giu prefix cu -> caller khong doi
CHUNK_GAP_SECONDS = 0.12
DEFAULT_CHUNK_CHARS = 240


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def get_engine():
    global _engine
    if _engine is None:
        log("Loading VieNeu-TTS v3 Turbo (GPU/bf16)...")
        _engine = VieNeuCloneEngine()
        log("VieNeu-TTS v3 Turbo loaded")
    return _engine


# ── I/O helpers (giu nguyen tu handler OmniVoice) ──────────────────────────

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
    except Exception as e:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
        log(f"WAV to MP3 conversion error: {e}")
        return False


def upload_to_r2(file_path: Path, job_id: str, r2_config: dict, content_type: str):
    """Up audio len Cloudflare R2, tra (presigned_url, object_key)."""
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
        object_key = f"{R2_KEY_PREFIX}/{job_id}_{uuid.uuid4().hex[:8]}{ext}"
        client.upload_file(
            str(file_path), r2_config["bucket_name"], object_key,
            ExtraArgs={"ContentType": content_type},
        )
        presigned_url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": r2_config["bucket_name"], "Key": object_key},
            ExpiresIn=7200,
        )
        return presigned_url, object_key
    except Exception as e:  # noqa: BLE001
        log(f"R2 upload error: {e}")
        return None, None


# ── Text chunking (giu tu OmniVoice; VieNeu infer con tu chia noi bo tiep) ──

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


# ── Duration / speed: atempo (giu pitch) tren wav cuoi ─────────────────────

def _atempo_factors(factor: float) -> list:
    """Chuoi atempo, moi tang trong [0.5, 2.0], tich = factor."""
    factor = max(0.25, min(4.0, factor))
    out = []
    while factor > 2.0:
        out.append(2.0); factor /= 2.0
    while factor < 0.5:
        out.append(0.5); factor /= 0.5
    out.append(factor)
    return out


def atempo_fit(wav_path: Path, out_path: Path, target: Optional[float], speed: float, actual: float) -> Path:
    # factor > 1 => nhanh hon (ngan lai). Uu tien ep duration (cue SRT), roi toi speed.
    factor = 1.0
    if target and target > 0 and actual and actual > 0:
        factor = actual / float(target)
    elif speed and speed > 0:
        factor = float(speed)
    if abs(factor - 1.0) < 1e-3:
        return wav_path
    chain = ",".join(f"atempo={f:.6f}" for f in _atempo_factors(factor))
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path), "-filter:a", chain, str(out_path)],
            capture_output=True, timeout=120, check=True,
        )
        return out_path
    except Exception as e:  # noqa: BLE001
        log(f"atempo error (bo qua, giu ban goc): {e}")
        return wav_path


# ── Handler ────────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})
    start_time = time.time()
    log(f"Job {job_id}: Starting VieNeu v3 Turbo")

    text = job_input.get("text")
    if not text or not str(text).strip():
        return {"error": "Missing required field: text"}

    output_format = job_input.get("output_format", "mp3")
    r2_config = job_input.get("r2")
    force_duration = job_input.get("duration")
    speed = float(job_input.get("speed", 1.0) or 1.0)
    denoise = bool(job_input.get("denoise", True))
    # Thanh dieu chinh giong (VieNeu native):
    style = job_input.get("style") or "tu_nhien"
    if style not in ("tu_nhien", "tin_tuc", "doc_truyen"):
        style = "tu_nhien"
    temperature = float(job_input.get("temperature", 0.8) or 0.8)   # bien hoa ngu dieu (chong "deu deu")
    top_p = float(job_input.get("top_p", 0.95) or 0.95)
    # ref_text / instruct / language / num_step / guidance_scale / t_shift: nhan nhung BO QUA

    work_dir = Path(tempfile.mkdtemp(prefix=f"vieneu_{job_id}_"))
    log(f"Working directory: {work_dir}")

    try:
        ref_path = None
        if job_input.get("ref_audio_url"):
            ref_path = work_dir / "ref_audio.wav"
            if not download_file(job_input["ref_audio_url"], ref_path):
                return {"error": "Failed to download reference audio"}
        elif job_input.get("ref_audio_base64"):
            ref_path = work_dir / "ref_audio.wav"
            if not decode_base64_file(job_input["ref_audio_base64"], ref_path):
                return {"error": "Failed to decode reference audio"}

        mode = "clone" if ref_path is not None else "auto"
        eng = get_engine()
        sr = eng.sample_rate
        slot = f"_job_{job_id}"   # slot enroll RIENG theo job (an toan neu concurrency > 1)

        if mode == "clone":
            eng.enroll(ref_path, slot, denoise=denoise)

        try:
            chunks = chunk_text(str(text))
            log(f"Job {job_id}: mode={mode}, {len(chunks)} chunk(s), generating...")
            gap = np.zeros(int(sr * CHUNK_GAP_SECONDS), dtype=np.float32)
            parts = []
            for i, ch in enumerate(chunks):
                wav_i = eng.synth_chunk(ch, mode, slot, style=style, temperature=temperature, top_p=top_p)
                if wav_i.size:
                    parts.append(wav_i)
                    if i < len(chunks) - 1:
                        parts.append(gap)
            if not parts:
                return {"error": "No audio generated"}
            wav = np.concatenate(parts) if len(parts) > 1 else parts[0]
        finally:
            if mode == "clone":
                eng.unenroll(slot)

        wav = match_loudness(wav)
        wav_path = work_dir / "output.wav"
        sf.write(str(wav_path), wav, sr)

        # Ep duration (cue SRT) va/hoac speed -> atempo (giu pitch)
        if (force_duration and float(force_duration) > 0) or (speed and speed != 1.0):
            stretched = work_dir / "output_fit.wav"
            wav_path = atempo_fit(wav_path, stretched,
                                  float(force_duration) if force_duration else None,
                                  speed, len(wav) / sr)

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
        log(f"Job {job_id}: generated in {elapsed:.1f}s ({duration:.2f}s audio)")

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

    except Exception as e:  # noqa: BLE001
        import traceback
        log(f"Handler exception: {e}")
        log(traceback.format_exc())
        return {"error": f"Internal error: {str(e)}"}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# RunPod serverless entry point
if __name__ == "__main__":
    log("Starting RunPod VieNeu v3 Turbo handler...")
    try:
        import torch
        if torch.cuda.is_available():
            log(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            log("WARNING: CUDA not available — clone yeu cau PyTorch/GPU!")
    except ImportError:
        log("Warning: torch not imported for CUDA check")

    # Nap + warmup NGAY luc khoi dong worker (khong lazy) -> FlashBoot snapshot process am,
    # request dau khong chiu phi nap model + capture CUDA graph.
    try:
        get_engine().warmup()
    except Exception as e:  # noqa: BLE001
        log(f"Init/warmup error (se thu lai o request dau): {e}")

    runpod.serverless.start({"handler": handler})
