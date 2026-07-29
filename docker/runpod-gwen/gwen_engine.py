"""Gwen-TTS clone engine wrapper cho RunPod worker.

Gwen-TTS = fine-tune tieng Viet cua Qwen3-TTS-0.6B (G-Group AI Lab, MIT). Clone giong
tu clip mau + transcript (ref_text). Nap model MOT LAN (bf16 GPU, attn sdpa -> KHONG can
flash-attn nen build nhanh, khong can nvcc) va expose API nho cho handler:
- clone(text, ref_audio_path, ref_text, language, temperature) -> np.float32 mono @ sample_rate
- warmup()
- match_loudness ve -11 dBFS (dong muc voi giong Kokoro/VieNeu).

API tham chieu (README gwen-tts):
  from qwen_tts import Qwen3TTSModel
  model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=torch.bfloat16,
                                        attn_implementation="sdpa")
  wavs, sr = model.generate_voice_clone(text=..., language="Vietnamese",
                                        ref_audio=<path>, ref_text=<transcript>, temperature=0.3, ...)
"""
import logging

import numpy as np

log = logging.getLogger("gwen-worker")

MODEL_ID = "g-group-ai-lab/gwen-tts-0.6B"
DEFAULT_LANGUAGE = "Vietnamese"

# Loudness parity: dong muc -11 dBFS voi OmniVoice/Kokoro/VieNeu (port tu vieneu_engine).
TARGET_RMS_DBFS = -11.0
PEAK_CEILING = 0.97


def match_loudness(audio):
    """Dua RMS (phan co tieng) ve TARGET_RMS_DBFS, co tran chong clip."""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    voiced = a[np.abs(a) > 0.01]
    if voiced.size == 0:
        return a
    rms = float(np.sqrt(np.mean(voiced.astype(np.float64) ** 2)))
    if rms <= 0:
        return a
    gain = (10.0 ** (TARGET_RMS_DBFS / 20.0)) / rms
    peak = float(np.max(np.abs(a)))
    if peak * gain > PEAK_CEILING:
        gain = PEAK_CEILING / peak
    return (a * gain).astype(np.float32)


def _to_mono_f32(wavs):
    """Chuan hoa output model (torch.Tensor | list | np.ndarray, mono/da kenh) -> np.float32 1D."""
    try:
        import torch
        if isinstance(wavs, torch.Tensor):
            wavs = wavs.detach().to("cpu", dtype=torch.float32).numpy()
    except Exception:  # noqa: BLE001
        pass
    if isinstance(wavs, (list, tuple)):
        parts = [_to_mono_f32(w) for w in wavs]
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    a = np.asarray(wavs, dtype=np.float32).squeeze()
    if a.ndim > 1:
        a = a.mean(axis=0)  # (C, T) -> mono
    return a.reshape(-1)


class GwenCloneEngine:
    def __init__(self):
        import torch
        from qwen_tts import Qwen3TTSModel

        self._torch = torch
        # sdpa: attention PyTorch native -> KHONG can flash-attn (build nhanh). Qwen3 ho tro sdpa.
        # bf16: yeu cau GPU Ampere+ (RunPod AMPERE_24 OK; HONG tren T4/RTX20).
        self._model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.sample_rate = int(getattr(self._model, "sample_rate", 24000) or 24000)
        log.info("Gwen-TTS (%s) loaded: sr=%d", MODEL_ID, self.sample_rate)

    def warmup(self, ref_audio_path=None, ref_text="Xin chào."):
        """Sinh 1 cau ngan de nap kernel/graph. Clone can 1 ref -> chi warmup khi co san ref."""
        if not ref_audio_path:
            log.info("Warmup bo qua (khong co ref); model da nap san khi khoi dong.")
            return
        try:
            self.clone("Xin chào.", ref_audio_path, ref_text, DEFAULT_LANGUAGE)
            log.info("Warmup done")
        except Exception as e:  # noqa: BLE001
            log.warning("Warmup failed (bo qua): %s", e)

    def clone(self, text, ref_audio_path, ref_text, language=DEFAULT_LANGUAGE, temperature=0.3):
        """Sinh 1 chunk clone -> np.float32 mono @ sample_rate.

        ref_text = transcript cua clip mau (Gwen/Qwen3-TTS CAN de canh acoustic prompt voi text).
        Neu thieu ref_text -> chuyen "" (chat luong kem hon; handler nen ASR bu neu can).
        """
        temp = float(temperature) if temperature and float(temperature) > 0 else 0.3
        wavs, sr = self._model.generate_voice_clone(
            text=str(text),
            language=language or DEFAULT_LANGUAGE,
            ref_audio=str(ref_audio_path),
            ref_text=(ref_text or "").strip(),
            temperature=temp,
            top_k=20,
            top_p=0.9,
            max_new_tokens=4096,
            repetition_penalty=2.0,
            subtalker_do_sample=True,
            subtalker_temperature=0.1,
            subtalker_top_k=20,
            subtalker_top_p=1.0,
        )
        if sr:
            self.sample_rate = int(sr)
        return _to_mono_f32(wavs)
