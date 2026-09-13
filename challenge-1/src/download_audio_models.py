"""
One-time setup: pull the faster-whisper AUDIO model weights to disk so
extract_transcriptions.py never needs network access to run.

    pip install faster-whisper
    small  -> used online (better accuracy, worth the download)
    tiny   -> used offline (small enough to ship, no network needed at inspection time)

See download_vision_language_models.py for the image-side equivalent (the
photo stamps and the photographed paper form).

Usage:
    python download_audio_models.py

Takes no arguments: one run fetches both small and tiny. Whatever is already
on disk is skipped, so re-running after an interrupted download costs only
the missing size.

Writes to:
    challenge-1/.models/audio/faster-whisper-<size>/
"""

import time
from pathlib import Path
from faster_whisper.utils import download_model
from logconf import setup
log = setup("download_audio_models")
AUDIO_DIR = Path(__file__).resolve().parent.parent / ".models" / "audio"


def size_on_disk(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def fetch(size):
    dest = AUDIO_DIR / f"faster-whisper-{size}"
    if dest.is_dir() and any(dest.iterdir()):
        log.info("%-8s already present, skipping (%.0f MB)", size, size_on_disk(dest) / 1e6)
        return 0.0

    log.info("%-8s downloading -> %s", size, dest)
    t0 = time.monotonic()
    try:
        download_model(size, output_dir=str(dest))
    except Exception as exc:
        log.error("%-8s failed: %s", size, exc)
        raise SystemExit(1)
    elapsed = time.monotonic() - t0

    mb = size_on_disk(dest) / 1e6
    log.info("%-8s done in %.1fs  %.0f MB  %.1f MB/s",
             size, elapsed, mb, mb / max(elapsed, 0.1))
    return elapsed


SIZES = ("small", "tiny")


def main():
    """Fetch both whisper sizes. Anything already on disk is skipped."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    for size in SIZES:
        fetch(size)
    log.info("%d model(s) in %.1fs, %.0f MB total in %s",
             len(SIZES), time.monotonic() - t0, size_on_disk(AUDIO_DIR) / 1e6, AUDIO_DIR)


if __name__ == "__main__":
    main()