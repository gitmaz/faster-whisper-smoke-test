"""Transcribe a short clip on the GPU and report device, timing, and text."""

import os
import sys
import time
from pathlib import Path

import ctranslate2
from faster_whisper import WhisperModel

AUDIO = Path(__file__).with_name("samples") / "jfk.flac"
MODEL = os.environ.get("WHISPER_MODEL", "small")
COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16")


def main() -> int:
    if not AUDIO.is_file():
        print(f"Missing audio file: {AUDIO}", file=sys.stderr)
        return 1

    cuda_devices = ctranslate2.get_cuda_device_count()
    print(f"ctranslate2 {ctranslate2.__version__}")
    print(f"CUDA devices visible to CTranslate2: {cuda_devices}")
    if cuda_devices < 1:
        print("No CUDA device found. GPU transcription cannot run.", file=sys.stderr)
        return 1

    print(f"Loading {MODEL} on cuda ({COMPUTE})...")
    load_started = time.perf_counter()
    model = WhisperModel(MODEL, device="cuda", compute_type=COMPUTE)
    load_seconds = time.perf_counter() - load_started
    print(f"Model loaded in {load_seconds:.2f}s")

    def transcribe():
        started = time.perf_counter()
        segments, info = model.transcribe(str(AUDIO), beam_size=5)
        lines = []
        for segment in segments:
            lines.append(
                f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}"
            )
        return lines, info, time.perf_counter() - started

    print(f"Transcribing {AUDIO.name}...")
    # The first GPU call pays for CUDA startup. Time a second pass.
    transcribe()
    lines, info, elapsed = transcribe()
    for line in lines:
        print(line)

    print()
    print(f"Language: {info.language} ({info.language_probability:.2%})")
    print(f"Audio duration: {info.duration:.2f}s")
    print(f"Transcription time (warmed up): {elapsed:.2f}s")
    if info.duration > 0:
        print(f"Realtime factor: {elapsed / info.duration:.3f}x (lower is faster)")
    if not lines:
        print("No text was produced.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
