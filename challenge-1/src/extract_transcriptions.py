"""
Step 1: transcribe every voice note in a municipality folder, once, to disk.

Everything downstream cites these transcripts, so each segment carries the
signals needed to decide whether a fact may be drawn from it.

Usage:
    python download_audio_models.py                          # once, fetches small + tiny
    python extract_transcriptions.py data/altamira            # standard, best accuracy
    python extract_transcriptions.py data/altamira --lite     # lite, on-device speed

Writes one file per model, so the two never overwrite each other:
    data/altamira/.cache/transcripts_standard.json
    data/altamira/.cache/transcripts_lite.json
"""

import argparse
import json
import logging
import math
from pathlib import Path

from faster_whisper import WhisperModel

from logconf import setup

log = setup("transcribe")

MODELS_DIR = Path(__file__).resolve().parent.parent / ".models" / "audio"

AUDIO_EXT = {".mp3", ".m4a", ".wav", ".ogg"}

MODEL_SIZE = {"standard": "small", "lite": "tiny"}

# Conditions the decoder on the vocabulary of this domain. Without it a small
# model over engine noise renders "igarape" and "leira" as phonetic guesses.
DOMAIN_HINT = (
    "Fiscalizacao ambiental no Para. Termos: hectare, igarape, vicinal, ramal, "
    "leira, APP, reserva legal, CAR, auto de infracao, auto de constatacao, "
    "embargo, supressao de vegetacao, trator de esteira, motosserra, "
    "patio de toras, coordenada, vertice, autuado, SEMMA."
)


def flag(seg):
    """Why this segment may not be trustworthy. Empty list means clean.

    no_speech      Whisper detected silence or engine noise.
    low_confidence Acoustically uncertain; numbers and names are unreliable.
    repetition     Decoder fell into a loop; text repeats itself.
    """

    reasons = []

    if seg.no_speech_prob > 0.6:
        reasons.append("no_speech")

    if seg.avg_logprob < -1.0:
        reasons.append("low_confidence")

    if seg.compression_ratio > 2.4:
        reasons.append("repetition")

    return reasons


def transcribe_one(model, path):
    segments, info = model.transcribe(
        str(path),
        language="pt",
        task="transcribe",
        vad_filter=True,                     # skip silence; big win outdoors
        initial_prompt=DOMAIN_HINT,
        condition_on_previous_text=False,    # one bad segment cannot poison the rest
    )

    out = []

    for s in segments:
        out.append({
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "text": s.text.strip(),
            "confidence": round(math.exp(s.avg_logprob), 3),
            "flags": flag(s),
        })

    return out, round(info.duration, 2)


def load_model(model_size):
    path = MODELS_DIR / f"faster-whisper-{model_size}"
    if not path.is_dir():
        log.error("missing %s -- run: python download_audio_models.py %s", path, model_size)
        raise SystemExit(1)
    log.debug("loading %s", path)
    return WhisperModel(str(path), device="cpu", compute_type="int8",
                        local_files_only=True)


def main(folder, lite=False, force=False):
    folder = Path(folder)
    audio_dir = folder / "audios"
    if not audio_dir.is_dir():
        log.error("no audios/ inside %s", folder)
        raise SystemExit(1)
    profile = "lite" if lite else "standard"
    model_size = MODEL_SIZE[profile]
    cache_path = folder / ".cache" / f"transcripts_{profile}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    files = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXT)
    # force re-reads every file and overwrites its entry rather than deleting
    # the cache first: a run that fails half way leaves the old answers intact.
    todo = files if force else [p for p in files if p.name not in cache]
    if not todo:
        log.info("%s: all %d files already in %s", folder.name, len(files), cache_path.name)
        return
    log.info("%s: whisper %s, %d of %d to transcribe", folder.name, profile, len(todo), len(files))
    model = load_model(model_size)
    for path in todo:
        segments, duration = transcribe_one(model, path)
        cache[path.name] = {
            "model": f"faster-whisper-{model_size}",
            "profile": profile,
            "duration": duration,
            "segments": segments,
        }
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        bad = sum(1 for s in segments if s["flags"])
        log.info("  %-24s %6.1fs  %2d segments  %d flagged",
                 path.name, duration, len(segments), bad)
    log.info("wrote %s", cache_path)
    flagged = [(n, s) for n in sorted(cache) for s in cache[n]["segments"] if s["flags"]]
    if flagged:
        log.warning("%d segments not safe to cite:", len(flagged))
        for name, s in flagged:
            log.warning("  %s @%.1f  %-28s %s",
                        name, s["start"], ",".join(s["flags"]), s["text"][:50])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("folder", nargs="?", default=".")
    p.add_argument("--lite", action="store_true",
                   help="use tiny instead of small: lower accuracy, smaller footprint")
    p.add_argument("--force", action="store_true",
                   help="re-transcribe every file, ignoring what is already cached")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    a = p.parse_args()
    if a.verbose:
        log.setLevel(logging.DEBUG)
    main(a.folder, lite=a.lite, force=a.force)
