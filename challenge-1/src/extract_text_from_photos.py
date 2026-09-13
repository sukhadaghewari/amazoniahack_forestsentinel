"""
Read the burned-in stamp on each field photo in photos/ - the only metadata
this pipeline can pull without knowing what the photo depicts:

    12 de mar de 2026 08:31:52
    3°18'13"S
    52°23'00"O
    SEMMA - ALTAMIRA

Filenames are bare numbers, so time and position are the only handle a later
step has to bind a photo to the voice note recorded beside it.

EXIF, where present, is exact and read first. The stamp is OCR, serving as
fallback and cross-check: agreement between two independent readings raises
confidence, disagreement surfaces and never resolves silently.

A value that could not be read is present with value: None - never absent,
never guessed - because a missing fact is recoverable and a confidently wrong
one is not.

pip install piexif paddleocr paddlepaddle
"""

import argparse
import json
import logging
import re
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import piexif
from PIL import Image

from facts import cache_path, fact, missing
from logconf import setup

log = setup("extract_photos")

PHOTO_EXT = {".jpg", ".jpeg", ".png"}

_PT_MONTHS = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}

_STAMP_DATE_RE = re.compile(
    r"(?:(?P<day1>\d{1,2})\s+de\s+(?P<mon>[a-zç]{3})\.?\s+de\s+(?P<year1>\d{4})"
    r"|(?P<day2>\d{1,2})/(?P<month2>\d{1,2})/(?P<year2>\d{4}))"
    r"\s+(?P<hour>\d{1,2})[:.](?P<minute>\d{2})(?:[:.](?P<second>\d{2}))?",
    re.I,
)

_DMS_SUFFIX_RE = re.compile(
    r'(\d+)\s*°\s*(\d+)\s*[\'′]\s*(\d+(?:[.,]\d+)?)\s*[\"″]?\s*([NSEOWL0])',
    re.I,
)

_DMS_PREFIX_RE = re.compile(
    r'([NSEOWL0])\s*(\d+)\s*°\s*(\d+)\s*[\'′]\s*(\d+(?:[.,]\d+)?)\s*[\"″]?',
    re.I,
)

_DMS_RE = re.compile(f"(?:{_DMS_SUFFIX_RE.pattern})|(?:{_DMS_PREFIX_RE.pattern})", re.I)
_FIRST_LETTER_RE = re.compile(r"[NSEOWL]", re.I)
_FIRST_DEGREE_RE = re.compile(r"\d+\s*°")
_DIR_SIGN = {"N": 1, "S": -1, "E": 1, "L": 1, "O": -1, "W": -1, "0": -1}
_COORD_TOL = 0.0005


def _dms_matches(text: str) -> list[tuple[str, str, str, str]]:
    """Return DMS readings using the convention detected in this text."""
    letter = _FIRST_LETTER_RE.search(text)
    degree = _FIRST_DEGREE_RE.search(text)

    if letter and (not degree or letter.start() < degree.start()):
        return [(deg, minutes, seconds, direction)
                for direction, deg, minutes, seconds in _DMS_PREFIX_RE.findall(text)]

    return _DMS_SUFFIX_RE.findall(text)


def _dms_to_decimal(deg: str, minutes: str, seconds: str, direction: str) -> float | None:
    sign = _DIR_SIGN.get(direction.upper())
    if sign is None:
        return None

    decimal = float(deg) + float(minutes) / 60 + float(seconds.replace(",", ".")) / 3600
    return round(decimal * sign, 5)


def parse_coordinate_pair(text: str) -> tuple[float, float] | None:
    """Find one N/S and one E/O/W/L reading -> (lat, lon)."""
    lat = lon = None

    for deg, minutes, seconds, direction in _dms_matches(text):
        value = _dms_to_decimal(deg, minutes, seconds, direction)
        if value is None:
            continue

        if direction.upper() in ("N", "S"):
            lat = value
        else:
            lon = value

    return (lat, lon) if lat is not None and lon is not None else None


def parse_stamp_timestamp(text: str) -> str | None:
    """Parse Portuguese or numeric date formats into ISO 8601."""
    match = _STAMP_DATE_RE.search(text)
    if not match:
        return None

    if match.group("mon"):
        day, year = match.group("day1"), match.group("year1")
        month = _PT_MONTHS.get(match.group("mon").lower())
    else:
        day, year = match.group("day2"), match.group("year2")
        month = int(match.group("month2"))

    if month is None:
        return None

    hour = match.group("hour")
    minute = match.group("minute")
    second = match.group("second") or "00"

    try:
        return datetime(int(year), month, int(day), int(hour), int(minute), int(second)).isoformat()
    except ValueError:
        return None


_STAMP_WIDTH_FRACTION = 0.35
_STAMP_TOP_FRACTION = 0.7
_STAMP_BOTTOM_FRACTION = 1.0

_WATERMARK_BAR_MAX_BRIGHTNESS = 50
_WATERMARK_BAR_MAX_STD = 20
_WATERMARK_BAR_CONFIRM_ROWS = 3
_WATERMARK_BAR_MARGIN_FRACTION = 0.03

_WATERMARK_WORDS = ("DOCU", "FICTÍCIO", "SINTÉTICOS", "VALIDADE")


def _watermark_bar_top(image: Image.Image) -> int:
    """Return watermark bar top, or image height if no bar is detected."""
    gray = np.array(image.convert("L"))
    h, w = gray.shape
    middle = gray[:, w // 3 : 2 * w // 3]

    confirmed = 0
    for y in range(h - 1, max(-1, h - 1 - _WATERMARK_BAR_CONFIRM_ROWS), -1):
        if middle[y].std() != 0.0:
            break
        confirmed += 1

    if confirmed < _WATERMARK_BAR_CONFIRM_ROWS:
        return h

    top = h
    for y in range(h - 1, -1, -1):
        row = middle[y]
        if row.mean() > _WATERMARK_BAR_MAX_BRIGHTNESS or row.std() > _WATERMARK_BAR_MAX_STD:
            break
        top = y

    return max(0, top - int(h * _WATERMARK_BAR_MARGIN_FRACTION))


_MODELS_DIR = Path(__file__).resolve().parent.parent / ".models" / "vision"

_DOWNLOAD_SCRIPT = "download_vision_language_models.py"


class _PPOCRProfile(NamedTuple):
    """One OCR tier: which weights to load, and where they sit on disk."""

    det_name: str
    rec_name: str
    dirname: str          # must match the paddleocr-<tier> dir the downloader writes


# lite/standard here are the same two tiers the downloader calls mobile/server,
# named to match the audio side's lite/standard rather than PaddleOCR's own words.
_PPOCR_PROFILES = {
    "lite": _PPOCRProfile(
        "PP-OCRv5_mobile_det", "latin_PP-OCRv5_mobile_rec", "paddleocr-mobile",
    ),
    "standard": _PPOCRProfile(
        "PP-OCRv5_server_det", "PP-OCRv5_server_rec", "paddleocr-server",
    ),
}

_DEFAULT_PROFILE = "standard"


def _stamp_crop_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Return bottom-left and bottom-right candidate stamp crops."""
    w, h = image.size
    top = int(h * _STAMP_TOP_FRACTION)
    bottom = min(int(h * _STAMP_BOTTOM_FRACTION), _watermark_bar_top(image))
    band_width = int(w * _STAMP_WIDTH_FRACTION)

    return [(0, top, band_width, bottom), (w - band_width, top, w, bottom)]


@lru_cache(maxsize=None)
def _get_ocr(profile: str):
    """Offline PaddleOCR instance - built once per profile, not once per photo."""
    det_name, rec_name, dirname = _PPOCR_PROFILES[profile]
    det_dir = _MODELS_DIR / dirname / "det"
    rec_dir = _MODELS_DIR / dirname / "rec"

    if not all((d / "inference.json").exists() for d in (det_dir, rec_dir)):
        log.error("PaddleOCR %s weights missing under %s - run: python %s",
                  profile, _MODELS_DIR / dirname, _DOWNLOAD_SCRIPT)
        raise SystemExit(1)

    from paddleocr import PaddleOCR

    log.info("loading PaddleOCR %s (%s / %s)", profile, det_name, rec_name)

    return PaddleOCR(
        text_detection_model_name=det_name,
        text_detection_model_dir=str(det_dir),
        text_recognition_model_name=rec_name,
        text_recognition_model_dir=str(rec_dir),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _merge_split_lines(
    detections: list[tuple[str, float, list[float]]],
) -> list[tuple[str, float]]:
    """Merge text boxes belonging to the same printed line."""

    def same_row(a, b):
        overlap = min(a[3], b[3]) - max(a[1], b[1])
        if overlap <= 0:
            return False

        min_height = min(a[3] - a[1], b[3] - b[1])
        return overlap / min_height > 0.5

    rows: list[list[tuple[str, float, list[float]]]] = []

    for det in sorted(detections, key=lambda d: d[2][1]):
        if rows and same_row(rows[-1][-1][2], det[2]):
            rows[-1].append(det)
        else:
            rows.append([det])

    lines = []
    for row in rows:
        row.sort(key=lambda d: d[2][0])
        lines.append((" ".join(d[0] for d in row), sum(d[1] for d in row) / len(row)))

    return lines


_STAMP_UPSCALE = 2
_MIN_CONFIDENCE = 0.6


def _ocr_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    profile: str,
) -> list[tuple[str, float]]:
    """OCR one candidate stamp crop."""
    crop = image.crop(box)
    crop = crop.resize((crop.width * _STAMP_UPSCALE, crop.height * _STAMP_UPSCALE), Image.LANCZOS)

    detections = []

    for res in _get_ocr(profile).predict(np.array(crop)):
        r = res.json["res"]
        detections.extend(zip(r["rec_texts"], r["rec_scores"], r["rec_boxes"]))

    detections = [d for d in detections if d[1] >= _MIN_CONFIDENCE]
    lines = _merge_split_lines(detections)

    return [(text, score) for text, score in lines
            if not any(word in text.upper() for word in _WATERMARK_WORDS)]


def read_stamp(path: Path, profile: str = _DEFAULT_PROFILE) -> dict | None:
    """OCR the burned-in corner stamp."""
    image = Image.open(path).convert("RGB")
    boxes = _stamp_crop_boxes(image)

    for box in boxes:
        kept = _ocr_crop(image, box, profile)
        if kept:
            lines, scores = zip(*kept)
            return {
                "lines": list(lines),
                "confidence": round(sum(scores) / len(scores), 3),
            }

    log.warning("%s: no stamp text found in either corner crop %s", path.name, boxes)
    return None


def _read_exif(path: Path) -> dict:
    """Best-effort EXIF read. Missing EXIF is normal."""
    try:
        exif = piexif.load(str(path))
    except Exception:
        return {}

    out = {}

    dt_raw = exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
    if dt_raw:
        dt_str = dt_raw.decode() if isinstance(dt_raw, bytes) else dt_raw

        try:
            out["timestamp"] = (
                datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S").isoformat(),
                dt_str,
            )
        except ValueError:
            pass

    gps = exif.get("GPS", {})
    lat = _gps_to_decimal(
        gps.get(piexif.GPSIFD.GPSLatitude),
        gps.get(piexif.GPSIFD.GPSLatitudeRef),
    )
    lon = _gps_to_decimal(
        gps.get(piexif.GPSIFD.GPSLongitude),
        gps.get(piexif.GPSIFD.GPSLongitudeRef),
    )

    if lat is not None and lon is not None:
        out["coordinate"] = (
            (round(lat, 5), round(lon, 5)),
            f"GPS {lat:.5f},{lon:.5f}",
        )

    return out


def _gps_to_decimal(dms, ref) -> float | None:
    if not dms or not ref:
        return None

    try:
        deg, minutes, seconds = (n / d for n, d in dms)
    except (TypeError, ZeroDivisionError):
        return None

    decimal = deg + minutes / 60 + seconds / 3600
    ref = ref.decode() if isinstance(ref, bytes) else ref

    return -decimal if ref.upper() in ("S", "W") else decimal


# EXIF is written by the camera itself; the stamp has to be read back off
# pixels, so it is trusted less, and least of all for the digits in a coordinate.
_EXIF_CONFIDENCE = 0.99
_STAMP_TIMESTAMP_CONFIDENCE = 0.85
_STAMP_COORDINATE_CONFIDENCE = 0.88
_STAMP_AGENCY_CONFIDENCE = 0.91

# Two independent readings that agree are worth slightly more than either alone,
# but never certainty - both can be wrong the same way. Two that disagree are
# worth less than either alone: one of them is wrong and nothing here says
# which, so the value ships below the confidence a single reading would have
# earned, with the disagreement recorded alongside it.
_AGREEMENT_BONUS = 0.05
_DISAGREEMENT_PENALTY = 0.10
_MAX_CONFIDENCE = 0.99


def _merge(
    rel: str,
    name: str,
    exif_value,
    exif_raw,
    exif_conf: float,
    stamp_value,
    stamp_raw,
    stamp_conf: float,
    close: Callable[[object, object], bool],
) -> dict:
    """Merge EXIF and stamp readings while preserving conflicts."""
    if exif_value is None and stamp_value is None:
        return missing()

    if exif_value is None:
        return fact(stamp_value, f"{rel}#stamp", stamp_conf, stamp_raw)

    if stamp_value is None:
        return fact(exif_value, f"{rel}#exif", exif_conf, exif_raw)

    agree = close(exif_value, stamp_value)
    confidence = (min(_MAX_CONFIDENCE, exif_conf + _AGREEMENT_BONUS) if agree
                  else max(0.0, exif_conf - _DISAGREEMENT_PENALTY))
    result = fact(exif_value, f"{rel}#exif", confidence, exif_raw)

    if not agree:
        result["conflict"] = {"exif": exif_value, "stamp": stamp_value}
        log.warning("%s: %s mismatch, exif=%s stamp=%s",
                    rel, name, exif_value, stamp_value)

    return result


def read_field_photo(path: Path, profile: str = _DEFAULT_PROFILE) -> dict:
    rel = f"photos/{path.name}"
    exif = _read_exif(path)
    stamp = read_stamp(path, profile)          # already warns when it reads nothing
    stamp_lines = stamp.get("lines", []) if stamp else []

    ts_raw = next((line for line in stamp_lines if _STAMP_DATE_RE.search(line)), None)
    stamp_ts = parse_stamp_timestamp(ts_raw) if ts_raw else None

    coord_lines = [line for line in stamp_lines if _DMS_RE.search(line)]
    stamp_coord = parse_coordinate_pair(" ".join(coord_lines)) if coord_lines else None
    coord_raw = " ".join(coord_lines) if coord_lines else None

    agency_raw = next(
        (line for line in reversed(stamp_lines)
         if line not in coord_lines and line != ts_raw),
        None,
    )

    if stamp_lines and ts_raw is None:
        log.warning("%s: no timestamp line matched: %r", rel, stamp_lines)

    if stamp_lines and not coord_lines:
        log.warning("%s: no coordinate line matched: %r", rel, stamp_lines)

    exif_ts, exif_ts_raw = exif.get("timestamp", (None, None))
    exif_coord, exif_coord_raw = exif.get("coordinate", (None, None))

    return {
        "timestamp": _merge(
            rel, "timestamp",
            exif_value=exif_ts, exif_raw=exif_ts_raw, exif_conf=_EXIF_CONFIDENCE,
            stamp_value=stamp_ts, stamp_raw=ts_raw,
            stamp_conf=_STAMP_TIMESTAMP_CONFIDENCE,
            close=lambda a, b: a == b,
        ),
        "coordinate": _merge(
            rel, "coordinate",
            exif_value=exif_coord, exif_raw=exif_coord_raw, exif_conf=_EXIF_CONFIDENCE,
            stamp_value=stamp_coord, stamp_raw=coord_raw,
            stamp_conf=_STAMP_COORDINATE_CONFIDENCE,
            close=lambda a, b: (
                abs(a[0] - b[0]) <= _COORD_TOL
                and abs(a[1] - b[1]) <= _COORD_TOL
            ),
        ),
        "agency": (
            fact(agency_raw, f"{rel}#stamp", _STAMP_AGENCY_CONFIDENCE, agency_raw)
            if agency_raw
            else missing()
        ),
    }


_FIELDS = ("timestamp", "coordinate", "agency")


def main(folder: Path, lite: bool = False, force: bool = False) -> None:
    """Read photos once and cache results separately per OCR profile."""
    profile = "lite" if lite else "standard"
    photos_dir = folder / "photos"
    if not photos_dir.is_dir():
        log.error("no photos/ inside %s", folder)
        raise SystemExit(1)

    cache_file = cache_path(folder, "photo_stamps", profile)
    cache = (
        json.loads(cache_file.read_text(encoding="utf-8"))
        if cache_file.exists()
        else {}
    )

    files = sorted(p for p in photos_dir.iterdir() if p.suffix.lower() in PHOTO_EXT)
    # force re-reads every photo and overwrites its entry rather than deleting
    # the cache first: a run that fails half way leaves the old answers intact.
    todo = files if force else [p for p in files if p.name not in cache]

    if not todo:
        log.info("%s: all %d photos already in %s",
                 folder.name, len(files), cache_file.name)
    else:
        log.info("%s: %d of %d photos to read", folder.name, len(todo), len(files))

        # Rewritten after every photo, not once at the end: OCR is slow enough
        # that a crash twenty photos in should not cost all twenty.
        for path in todo:
            cache[path.name] = read_field_photo(path, profile)
            cache_file.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        log.info("wrote %s", cache_file)

    for name in sorted(cache):
        result = cache[name]
        status = {
            field: "ok" if result[field]["value"] is not None else "miss"
            for field in _FIELDS
        }
        log.info("%-14s timestamp=%-4s coordinate=%-4s agency=%-4s",
                 name, status["timestamp"], status["coordinate"], status["agency"])

    for field in _FIELDS:
        got = sum(1 for result in cache.values() if result[field]["value"] is not None)
        log.info("%s: %d/%d photos readable", field, got, len(cache))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("folder", nargs="?", default=".")
    parser.add_argument("--lite", action="store_true",
                         help="use mobile OCR weights instead of server: lower accuracy, smaller footprint")
    parser.add_argument("--force", action="store_true",
                         help="re-read every photo, ignoring what is already cached")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()
    if args.verbose:
        log.setLevel(logging.DEBUG)

    main(Path(args.folder), lite=args.lite, force=args.force)