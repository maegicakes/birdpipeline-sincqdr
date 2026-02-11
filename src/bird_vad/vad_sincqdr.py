from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio


@dataclass(frozen=True)
class SincQDRSettings:
    checkpoint_path: str
    device: str = "cpu"

    # chunking params
    sample_rate: int = 16000
    window_duration: float = 0.63
    step_size: float = 0.08
    batch_size: int = 8

    threshold: float = 0.5
    min_speech_ms: int = 200
    min_silence_ms: int = 200

    # model arch
    patch_size: int = 8
    in_ch: int = 1
    c1: int = 32
    c2: int = 64
    num_classes: int = 2
    use_bn: bool = True


def _chunk_waveform(waveform: torch.Tensor, *, duration: float, step_size: float, sample_rate: int) -> torch.Tensor:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    chunk_samples = int(duration * sample_rate)
    step_samples = int(step_size * sample_rate)
    total = waveform.size(1)

    chunks = []
    start = 0
    while start + chunk_samples <= total:
        end = start + chunk_samples
        chunks.append(waveform[:, start:end])
        start += step_samples

    if not chunks:
        return torch.zeros((0, 1, chunk_samples), dtype=waveform.dtype)

    return torch.stack(chunks, dim=0)


def _merge_close(intervals: List[Tuple[float, float]], max_gap_s: float) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + max_gap_s:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _drop_short(intervals: List[Tuple[float, float]], min_len_s: float) -> List[Tuple[float, float]]:
    return [(s, e) for (s, e) in intervals if (e - s) >= min_len_s]


def run_sincqdr_vad(audio_path: str | Path, settings: SincQDRSettings) -> List[Tuple[float, float]]:
    audio_path = Path(audio_path)

    from model.sincqdr import SincQDR

    device = torch.device(settings.device)

    import numpy as np
    import soundfile as sf

    audio_np, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)  # shape [T, C]
    audio_mono = audio_np.mean(axis=1)  # [T]

    # torchaudio.load -> torchcodec
    if sr != settings.sample_rate:
        x_old = np.linspace(0.0, 1.0, num=audio_mono.shape[0], endpoint=False)
        new_len = int(round(audio_mono.shape[0] * (settings.sample_rate / sr)))
        x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        audio_mono = np.interp(x_new, x_old, audio_mono).astype(np.float32)
        sr = settings.sample_rate

    waveform = torch.from_numpy(audio_mono).unsqueeze(0)  # [1, T]


    # waveform, sr = torchaudio.load(str(audio_path))
    # waveform = waveform.mean(dim=0, keepdim=True)  # mono [1,T]
    # if sr != settings.sample_rate:
    #     waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=settings.sample_rate)

    chunks = _chunk_waveform(
        waveform,
        duration=settings.window_duration,
        step_size=settings.step_size,
        sample_rate=settings.sample_rate,
    )
    if chunks.size(0) == 0:
        return []

    model = SincQDR(
        settings.in_ch,
        settings.c1,
        settings.c2,
        settings.patch_size,
        settings.num_classes,
        settings.use_bn,
    ).to(device)

    ckpt = torch.load(settings.checkpoint_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    if isinstance(state, dict):
        cleaned = {}
        for k, v in state.items():
            nk = k[len("model."):] if k.startswith("model.") else k
            cleaned[nk] = v
        state = cleaned

    model.load_state_dict(state, strict=False)
    model.eval()

    outs = []
    with torch.no_grad():
        for i in range(0, chunks.size(0), settings.batch_size):
            batch = chunks[i:i + settings.batch_size].to(device)
            out = model(batch)
            outs.append(out.detach().cpu())

    outputs = torch.cat(outs, dim=0)
    probs = torch.sigmoid(outputs).flatten()

    windows: List[Tuple[float, float]] = []
    for i, p in enumerate(probs.tolist()):
        if float(p) >= settings.threshold:
            s = i * settings.step_size
            e = s + settings.window_duration
            windows.append((s, e))

    intervals = _merge_close(windows, max_gap_s=0.0)
    intervals = _merge_close(intervals, max_gap_s=settings.min_silence_ms / 1000.0)
    intervals = _drop_short(intervals, min_len_s=settings.min_speech_ms / 1000.0)

    return intervals
