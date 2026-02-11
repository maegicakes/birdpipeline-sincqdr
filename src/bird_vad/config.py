from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _get_env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise ValueError(f"Missing required env var: {name}")
    return v


def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError as e:
        raise ValueError(f"{name} must be an int, got {v!r}") from e


def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be boolean-like, got {v!r}")


@dataclass(frozen=True)
class RecorderConfig:
    arecord_device: str

    rate_hz: int         # 44100
    channels: int        # 2
    duration_s: int      # 60, 1min clips

    audio_dir: Path


@dataclass(frozen=True)
class UploadConfig:
    enabled: bool
    s3_bucket: str
    s3_prefix: str
    s3_endpoint: Optional[str]
    aws_region: str

    workers: int
    delete_after_upload: bool


@dataclass(frozen=True)
class VadConfig:
    threshold: float
    min_speech_ms: int
    min_silence_ms: int
    results_dir: Path


@dataclass(frozen=True)
class Config:
    device_id: str
    recorder: RecorderConfig
    vad: VadConfig
    upload: UploadConfig


def load_config(env_file: str = ".env") -> Config:
    """
    Loads environment variables (optionally from .env) using names aligned with:
      - bird-files-main/record/record_upload.py
      - bird-files-main/record/upload_to_s3.py
    """
    load_dotenv(env_file, override=False)

    device_id = os.getenv("DEVICE_ID") or os.uname().nodename

    # recorder
    arecord_device = os.getenv("ARECORD_DEVICE", "plughw:2,0")

    rate_hz = _get_int("ARECORD_RATE", 44100)
    channels = _get_int("ARECORD_CHANNELS", 2)
    duration_s = _get_int("ARECORD_DURATION", 60)

    audio_dir = Path(_get_env("AUDIO_DIR", "./data_temp/Audios")).expanduser().resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)

    recorder = RecorderConfig(
        arecord_device=arecord_device,
        rate_hz=rate_hz,
        channels=channels,
        duration_s=duration_s,
        audio_dir=audio_dir,
    )

    # output
    results_dir = Path(_get_env("RESULTS_DIR", "./data_temp/VAD_Results")).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    threshold = float(os.getenv("VAD_THRESHOLD", "0.5"))
    min_speech_ms = _get_int("MIN_SPEECH_MS", 200)
    min_silence_ms = _get_int("MIN_SILENCE_MS", 200)

    vad = VadConfig(
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        results_dir=results_dir,
    )

    upload_enabled = _get_bool("UPLOAD_ENABLED", True)

    # S3_BUCKET, S3_PREFIX, S3_ENDPOINT, AWS_REGION/AWS_DEFAULT_REGION
    s3_bucket = os.getenv("S3_BUCKET", "")
    s3_prefix = os.getenv("S3_PREFIX", "")
    s3_endpoint = os.getenv("S3_ENDPOINT")  
    aws_region = (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or ("auto" if s3_endpoint else "us-east-1")
    )

    workers = _get_int("UPLOAD_WORKERS", 4)
    delete_after = _get_bool("UPLOAD_DELETE", False)

    if upload_enabled and not s3_bucket:
        raise ValueError("UPLOAD_ENABLED=1 but S3_BUCKET is not set (bird-files uploader requires it).")

    upload = UploadConfig(
        enabled=upload_enabled,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        s3_endpoint=s3_endpoint,
        aws_region=aws_region,
        workers=workers,
        delete_after_upload=delete_after,
    )

    return Config(device_id=device_id, recorder=recorder, vad=vad, upload=upload)
