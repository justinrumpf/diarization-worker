"""RunPod serverless handler for speech diarization.

Same model stack as examples/speech-diarization (faster-whisper large-v3 +
pyannote speaker-diarization-3.1), repackaged as a serverless worker so the
zendesk-hud app can spin it up on demand and let it scale back to zero.

Input (event["input"]):
  {"url": "<audio url>", "headers": {"Authorization": "Bearer …"}}   # RingCentral
  {"base64": "<b64 audio>", "format": "mp3"}                          # inline
  optional: id, num_speakers, min_speakers, max_speakers, language, beam_size

Output:
  {"language", "language_probability", "duration", "num_speakers",
   "turns": [{"speaker","start","end","text"}], "id"?}

Env: HF_TOKEN (required, gated pyannote), WHISPER_MODEL, DEVICE, COMPUTE_TYPE.
"""

import base64
import os
import tempfile
from pathlib import Path

import av
import numpy as np
import requests
import runpod
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as DiarizationPipeline

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = os.environ.get("DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

_whisper = None
_diar = None


def get_whisper():
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _whisper


def get_diar():
    global _diar
    if _diar is None:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN required for pyannote; accept terms at "
                               "huggingface.co/pyannote/speaker-diarization-3.1")
        _diar = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
        if DEVICE == "cuda":
            _diar.to(torch.device("cuda"))
    return _diar


def _load_audio(path):
    container = av.open(path)
    stream = container.streams.audio[0]
    sr = stream.rate or stream.codec_context.sample_rate
    frames = [f.to_ndarray() for f in container.decode(audio=0)]
    container.close()
    audio = np.concatenate(frames, axis=1)
    wav = torch.from_numpy(audio).float()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def _resolve_audio(data):
    """Return (path, is_temp). Supports url (+optional headers) / base64 / file_path."""
    if "url" in data:
        headers = data.get("headers") or {}
        resp = requests.get(data["url"], headers=headers, timeout=300, stream=True)
        resp.raise_for_status()
        suffix = Path(data["url"].split("?")[0]).suffix or ".mp3"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        for chunk in resp.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp.close()
        return tmp.name, True
    if "base64" in data:
        suffix = data.get("format", ".mp3")
        suffix = suffix if suffix.startswith(".") else "." + suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(base64.b64decode(data["base64"]))
        tmp.close()
        return tmp.name, True
    if "file_path" in data:
        return data["file_path"], False
    raise ValueError("input must include 'url', 'base64', or 'file_path'")


def _turns(segments, diarization):
    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({"start": w.start, "end": w.end, "word": w.word.strip()})
    if not words:   # no word timestamps → segment-level
        out = []
        for seg in segments:
            mid = (seg.start + seg.end) / 2
            best, bov = "UNKNOWN", 0.0
            for turn, _, spk in diarization.itertracks(yield_label=True):
                ov = max(0.0, min(seg.end, turn.end) - max(seg.start, turn.start))
                if ov > bov:
                    bov, best = ov, spk
            out.append({"speaker": best, "start": round(seg.start, 3),
                        "end": round(seg.end, 3), "text": seg.text.strip()})
        return out
    for word in words:
        best, bov = "UNKNOWN", 0.0
        for turn, _, spk in diarization.itertracks(yield_label=True):
            ov = max(0.0, min(word["end"], turn.end) - max(word["start"], turn.start))
            if ov > bov:
                bov, best = ov, spk
        word["speaker"] = best
    turns, cur, cw, cs = [], None, [], 0.0
    for word in words:
        if word["speaker"] != cur:
            if cw:
                turns.append({"speaker": cur, "start": round(cs, 3),
                              "end": round(cw[-1]["end"], 3),
                              "text": " ".join(w["word"] for w in cw)})
            cur, cw, cs = word["speaker"], [word], word["start"]
        else:
            cw.append(word)
    if cw:
        turns.append({"speaker": cur, "start": round(cs, 3), "end": round(cw[-1]["end"], 3),
                      "text": " ".join(w["word"] for w in cw)})
    return turns


def process(data):
    path, is_temp = _resolve_audio(data)
    try:
        seg_gen, info = get_whisper().transcribe(
            path, beam_size=data.get("beam_size", 5), word_timestamps=True,
            vad_filter=True, language=data.get("language"))
        segments = list(seg_gen)
        dk = {k: data[k] for k in ("num_speakers", "min_speakers", "max_speakers") if k in data}
        wav, sr = _load_audio(path)
        diar_out = get_diar()({"waveform": wav, "sample_rate": sr}, **dk)
        diar = getattr(diar_out, "speaker_diarization", diar_out)
        turns = _turns(segments, diar)
        return {"language": info.language,
                "language_probability": round(info.language_probability, 4),
                "duration": round(info.duration, 3),
                "num_speakers": len({t["speaker"] for t in turns}),
                "turns": turns}
    finally:
        if is_temp and os.path.exists(path):
            os.unlink(path)


def handler(event):
    data = event.get("input") or {}
    try:
        result = process(data)
        if "id" in data:
            result["id"] = data["id"]
        return result
    except Exception as exc:  # noqa: BLE001 — surface as job error
        return {"error": f"{type(exc).__name__}: {exc}", "id": data.get("id")}


runpod.serverless.start({"handler": handler})
