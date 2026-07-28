"""VieNeu-TTS v3 Turbo clone engine wrapper cho RunPod worker.

Nap engine v3 Turbo PyTorch (bf16 tren GPU) MOT LAN va expose API nho cho handler:
- enroll giong tham chieu MOT LAN (add_voice -> speaker_emb + ref_codes) roi tai dung,
- synth 1 chunk co retry chong "runaway" (VieNeu doi khi sinh lan man ra tieng la),
- match_loudness ve -11 dBFS (dong muc voi giong Kokoro/VieNeu tren VPS).

API tham chieu tu source vieneu (v3turbo.py): Vieneu(device="cuda") -> V3TurboVieNeuTTS;
  tts.add_voice(name, ref_wav, denoise=True); tts.infer(text, voice=name, style=..., apply_watermark=False)
  -> np.float32 @ 48kHz. KHONG can ref_text.
LUU Y 3.2.3: doc chinh thuc dung kwarg `precision` (thay `dtype` cua 3.0.12) va noi clone can
engine PyTorch; tren GPU `Vieneu(device="cuda")` da chon PyTorch+bf16. Verify lai khi build.
"""
import logging

import numpy as np

log = logging.getLogger("vieneu-worker")

# Loudness parity: do that 2026-07-24 OmniVoice/Kokoro ~ -11 dBFS. Port tu tts-service/app.py.
TARGET_RMS_DBFS = -11.0
PEAK_CEILING = 0.97
MAX_SYNTH_ATTEMPTS = 4
DEFAULT_VOICE = "Trúc Ly"      # mode="auto": giong nu trung tinh (fallback ve default cua model neu thieu)
DEFAULT_STYLE = "doc_truyen"   # tu_nhien | tin_tuc | doc_truyen (chot doc_truyen: bot deu deu)


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


def expected_seconds(text):
    """Uoc thoi luong doc sach (khong runaway). Tieng Viet ~13 char/s -> ~0.08 s/char."""
    return max(0.6, 0.08 * len(text) + 0.6)


def is_anomalous(duration, text):
    """Co lan man/dut giua chung khong? > 2.5x hoac < 0.35x ky vong = bat thuong."""
    exp = expected_seconds(text)
    return duration > exp * 2.5 or duration < exp * 0.35


class VieNeuCloneEngine:
    def __init__(self):
        from vieneu import Vieneu

        # GPU worker: device="cuda" -> PyTorch v3 Turbo (bf16). Bat buoc GPU: ban CPU
        # torch-free (ONNX-lite) KHONG clone duoc.
        try:
            self._tts = Vieneu(device="cuda")
        except TypeError:
            # Phong lech kwarg giua ban vieneu (3.0.12 dung `device`/`dtype`, 3.2.3 co the khac):
            # tren GPU box `Vieneu()` auto van chon PyTorch+cuda.
            self._tts = Vieneu()
        backend = getattr(self._tts, "backend", None)
        if backend == "onnx":
            raise RuntimeError(
                "VieNeu dang o backend ONNX-lite (KHONG clone duoc). Worker nay can GPU/PyTorch — "
                "kiem tra CUDA co san trong container."
            )
        self.sample_rate = int(getattr(self._tts, "sample_rate", 48000))
        names = [vid for _label, vid in self._tts.list_preset_voices()]
        self.default_voice = DEFAULT_VOICE if DEFAULT_VOICE in names else (names[0] if names else None)
        log.info("VieNeu v3 Turbo ready: sr=%d, %d preset voices, default=%s",
                 self.sample_rate, len(names), self.default_voice)

    def warmup(self):
        """Sinh 1 cau ngan luc khoi dong -> nap kernel/graph, request dau khong chiu phi."""
        try:
            self._tts.infer("Xin chào.", voice=self.default_voice, style=DEFAULT_STYLE, apply_watermark=False)
            log.info("Warmup done")
        except Exception as e:  # noqa: BLE001
            log.warning("Warmup failed (bo qua): %s", e)

    def enroll(self, ref_path, slot, denoise=True):
        """Encode speaker_emb + ref_codes MOT LAN duoi `slot` RIENG theo job; moi chunk tai dung.

        Slot rieng theo job (khong dung chung `_job`) de an toan neu endpoint bat concurrency
        >1 job/worker tren singleton dung chung.
        """
        self._tts.add_voice(slot, str(ref_path), denoise=denoise)

    def unenroll(self, slot):
        try:
            self._tts.remove_voice(slot)
        except Exception:  # noqa: BLE001
            pass

    def synth_chunk(self, text, mode, slot=None, style=DEFAULT_STYLE, temperature=0.7, top_p=0.95, silence_p=0.15):
        """Sinh 1 chunk -> np.float32 @ sample_rate, co retry chong runaway.

        style: tu_nhien | tin_tuc | doc_truyen (kieu doc). temperature: bien hoa ngu dieu
        (cao = it "deu deu" hon nhung de bat on hon). top_p: nucleus sampling. silence_p: khoang
        nghi giua cac cau con ben trong (Khoang nghi slider). Moi lan bat thuong -> giu ban it lech nhat.
        """
        voice = slot if (mode == "clone" and slot) else self.default_voice
        best = None
        best_gap = float("inf")
        for _ in range(MAX_SYNTH_ATTEMPTS):
            cand = self._tts.infer(text, voice=voice, style=style, temperature=temperature,
                                   top_p=top_p, silence_p=silence_p, apply_watermark=False)
            cand = np.asarray(cand, dtype=np.float32).reshape(-1)
            if cand.size == 0:
                continue
            dur = cand.size / self.sample_rate
            if not is_anomalous(dur, text):
                return cand
            gap = abs(dur - expected_seconds(text))
            if gap < best_gap:
                best, best_gap = cand, gap
        return best if best is not None else np.zeros(0, dtype=np.float32)
