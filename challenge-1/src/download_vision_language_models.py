"""
One-time setup: pull every local model weight the pipeline's image steps need,
so they never require network access, or a paid API, to run. Two models for
two different jobs - they are not interchangeable. See
download_audio_models.py for the audio-side equivalent (voice notes).

PaddleOCR PP-OCRv5 - photos/, 13-165 MB, runs on a laptop CPU
    Reads the four-line stamp the field app burns into every photo
    (timestamp, coordinates, agency) that extract_text_from_photos.py
    cross-checks against EXIF. A 13 MB specialist beats a 3 GB VLM here
    because the stamp isn't a form to interpret - it's four fixed lines at a
    known position, cropped before OCR even runs (see read_stamp in
    extract_text_from_photos.py).
    General-purpose OCR (Tesseract, EasyOCR) was tried first and struggled
    specifically with the DMS degree/minute/second marks; PP-OCRv5's Latin
    recognizer reads them cleanly.

    Two tiers, same split as the audio side's small/tiny:
        mobile  det + latin_rec   ~13 MB   ships in the field bundle (--lite)
        server  det + rec         ~165 MB  default; more robust on a bad
                                            frame (blur, glare, odd angle).
                                            No Latin-specific server
                                            recognizer exists yet, so this
                                            tier uses PP-OCRv5's general
                                            server rec model instead - still
                                            an upgrade on detection, roughly
                                            even on recognition.

Qwen3-VL 4B - ~3 GB, runs on a laptop CPU
    Two jobs, both reading rather than writing: pulling named values out of
    the free text in field-notes.md (extract_from_field_notes.py) and
    transcribing the photographed paper form section by section
    (extract_from_final_report.py). Apache-2.0, and currently one of the
    strongest open-weight families at OCR and document structure - trained
    across 32 languages (Portuguese included) with explicit robustness to
    blur, tilt and low light, which is what a phone photo of a form taken
    outdoors looks like. It still reads handwriting worse than a person; that
    is what the per-field confidence score is for. A low score means "needs a
    human", not noise to ignore.

    The 8B tier is deliberately not offered here. It roughly doubles the
    footprint for a modest gain and does not fit the hardware this has to run
    on. This pipeline targets grassroots deployment, not a datacenter.

    Nothing in this pipeline asks a model to write report prose: drafting is
    deterministic from the facts (see generate_report.py). So no text-only
    model is downloaded - a value that is not physically in the evidence has
    no decoding step it could be invented in.

Why Ollama rather than raw GGUF + llama-cpp-python: Qwen3-VL support in the
official bindings is not there yet. Ollama is one actively-maintained binary,
ships the official weights, manages quantization, and speaks a plain local
HTTP API.

Requires the Ollama daemon installed separately for the VLM half - this script
pulls weights, it does not install Ollama: https://ollama.com/download

Usage:
    pip install ollama huggingface_hub paddleocr paddlepaddle
    python download_vision_language_models.py

Takes no arguments: one run fetches everything - both PaddleOCR tiers and
Qwen3-VL. Whatever is already on disk is skipped, so re-running after an
interrupted or partial download costs only the missing pieces.

Storage is split by ecosystem: PaddleOCR lands in challenge-1/.models/vision/,
alongside .models/audio/ for faster-whisper. Ollama insists on its own store
(~/.ollama/models) and 3 GB does not belong in a repo anyway; to colocate it,
export OLLAMA_MODELS before starting the daemon.
"""

import shutil
import time
from pathlib import Path

from logconf import setup

log = setup("download_models")

VISION_DIR = Path(__file__).resolve().parent.parent / ".models" / "vision"

VLM_TAG = "qwen3-vl:4b"


def size_mb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


# PP-OCRv6 exists with a smaller detector as of writing, but PP-OCRv5's Latin
# recognizer is the one actually verified against this project's stamp
# photos (see extract_text_from_photos.py's read_stamp) - not swapped in
# without the same check.
PPOCR_REPOS = {
    "mobile": {"det": "PaddlePaddle/PP-OCRv5_mobile_det", "rec": "PaddlePaddle/latin_PP-OCRv5_mobile_rec"},
    "server": {"det": "PaddlePaddle/PP-OCRv5_server_det", "rec": "PaddlePaddle/PP-OCRv5_server_rec"},
}
PPOCR_PATTERNS = ["*.json", "*.yml", "*.pdiparams"]


def fetch_ocr(tier="mobile"):
    from huggingface_hub import snapshot_download

    dest = VISION_DIR / f"paddleocr-{tier}"
    for part, repo in PPOCR_REPOS[tier].items():
        part_dest = dest / part
        label = f"ocr-{tier}/{part}"
        if part_dest.is_dir() and (part_dest / "inference.json").exists():
            log.info("%-24s already present, skipping (%.0f MB)", label, size_mb(part_dest))
            continue

        log.info("%-24s downloading -> %s", label, part_dest)
        t0 = time.monotonic()
        try:
            snapshot_download(repo, local_dir=str(part_dest), allow_patterns=PPOCR_PATTERNS)
        except Exception as exc:
            log.error("%s download failed: %s", repo, exc)
            raise SystemExit(1)
        elapsed = time.monotonic() - t0
        mb = size_mb(part_dest)
        log.info("%-24s done in %.1fs  %.0f MB  %.1f MB/s",
                 label, elapsed, mb, mb / max(elapsed, 0.1))


def _ollama_size_mb(model):
    import ollama
    for m in ollama.list().models:
        if m.model in (model, f"{model}:latest"):
            return m.size / 1e6
    return None


def fetch_vlm(tag=VLM_TAG):
    import ollama

    if not shutil.which("ollama"):
        log.error("ollama not installed")
        raise SystemExit(
            "ollama is not installed. Get it from https://ollama.com/download, "
            "then re-run this script."
        )

    try:
        present = _ollama_size_mb(tag)
    except Exception as exc:
        log.error("could not reach the Ollama service: %s", exc)
        raise SystemExit(
            f"couldn't reach the Ollama service ({exc}). Is it running? "
            "Open the Ollama app, or run `ollama serve` in another terminal."
        )

    if present is not None:
        log.info("%-24s already present, skipping (%.0f MB)", tag, present)
        return

    log.info("%-24s pulling...", tag)
    t0 = time.monotonic()
    last = None
    for update in ollama.pull(tag, stream=True):
        if update.status != last:
            log.debug("  %s", update.status)
            last = update.status
    elapsed = time.monotonic() - t0
    mb = _ollama_size_mb(tag) or 0.0
    log.info("%-24s done in %.1fs  %.0f MB  %.1f MB/s",
             tag, elapsed, mb, mb / max(elapsed, 0.1))


def main():
    """Fetch every model the pipeline can use. Anything already on disk is skipped."""
    VISION_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()

    fetch_ocr("mobile")
    fetch_ocr("server")

    # Last, because it is the only step needing a separate daemon installed:
    # if Ollama is missing, everything above is already on disk.
    fetch_vlm()

    log.info("all models in %.1fs", time.monotonic() - t0)


if __name__ == "__main__":
    main()
