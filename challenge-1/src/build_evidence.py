"""
Consolidate every extractor's cache into one evidence file - the single
artefact the report draft is written from, and the only place an auditor has
to look to ask "who says so?".

Each extractor knows one modality and nothing else: photo stamps do not know
what the voice notes said, the occurrence summary does not know what the
inspector typed. A claim is only worth what its evidence is worth, so this
step puts every claim about the same thing side by side and asks whether the
modalities agree.

One fact is one claim about one thing:

    id          stable handle the report template refers to
    key         what is being claimed ("car_registry", "operator_name")
    type        shape of value - person, area, coordinate, document...
    value       the claim itself
    status      observed   at least one source states it directly
                inferred   computed from other facts; see derivation
                conflicting sources disagree; every reading is kept
                missing    the report needs it and no source has it
    confidence  derived, never asserted - see _score
    sources     every passage backing it, with modality and locator

Nothing here invents a value. A field the report needs and the evidence does
not support comes out as status "missing" with an empty source list, which is
the honest answer and the one a human can act on.

Reads every extractor's intermediate from data/<case>/.cache/ and writes:
    data/<case>/output/evidence.json

Usage:
    python build_evidence.py data/altamira
"""

import argparse
import json
import re
from pathlib import Path

from logconf import setup

log = setup("build_evidence")

# How far a single reading of each kind is trusted before corroboration.
# Ordered by how much interpretation sits between the source and the value:
# a parsed text field is read literally, a stamp has been through OCR, a
# transcript through speech recognition, an LLM extraction through both.
#
# Public, with the two adjustments below, because the UI explains confidence to
# a reader by printing these numbers. A legend that restated them by hand would
# eventually describe a scorer that no longer works this way.
MODALITY_CONFIDENCE = {
    "occurrence_summary": 0.99,
    "exif": 0.97,
    "photo_stamp": 0.85,
    "field_note": 0.80,
    "audio": 0.62,
    "llm_extraction": 0.68,
}

CORROBORATION_BONUS = 0.05      # per independent source beyond the first
FLAGGED_SEGMENT_PENALTY = 0.15  # cites a transcript segment Whisper flagged
_MAX_CONFIDENCE = 0.99

# Report fields with no source in the field evidence at all. They are issued
# later by the office - the fine and embargo come out of a board meeting five
# days after the inspection - so they are carried as explicitly missing rather
# than quietly absent, and a drafter can see what still needs a human.
#
# (pt-BR, English) per field: the Portuguese one is what a Brazilian reader
# sees, and the report is a Brazilian legal instrument - a reason written only
# in English cannot be shown next to an empty field in it.
_KNOWN_ABSENT = {
    "report_number": ("atribuído quando o relatório é protocolado, após a fiscalização",
                      "assigned when the report is filed, after the inspection"),
    "infraction_notice_number": ("emitido na reunião da junta de penalidades, 17/03",
                                 "issued at the penalty board meeting, 17/03"),
    "embargo_notice_number": ("emitido na reunião da junta de penalidades, 17/03",
                              "issued at the penalty board meeting, 17/03"),
    "inspector_full_name": ("apenas o primeiro nome é falado ou digitado em campo",
                            "only the first name is ever spoken or typed"),
    "team_full_names": ("apenas os primeiros nomes são falados ou digitados em campo",
                        "only first names are ever spoken or typed"),
}

_LOCATOR_RE = re.compile(r"^(?P<file>[^@#]+)[@#](?P<locator>.+)$")


def _parse_source(source: str, text: str | None = None) -> dict:
    """Turn an extractor's source string into a structured, auditable citation.

    Three formats reach this, one per extractor:
        photos/01.jpg#stamp                        photo, OCR'd stamp region
        field-notes.md@08:34                       typed note, by its timestamp
        04.mp3@13.14s                              voice note, by segment start
        .../occurrence-summary.txt#Área estimada   app export, by field name
    """
    match = _LOCATOR_RE.match(source)
    if not match:
        return {"modality": "unknown", "file": source, "text": text}

    file, locator = match["file"], match["locator"]
    suffix = Path(file).suffix.lower()

    if suffix in {".jpg", ".jpeg", ".png"}:
        modality = "exif" if locator == "exif" else "photo_stamp"
        citation = {"modality": modality, "file": file, "region": locator}
    elif suffix == ".mp3":
        modality = "audio"
        citation = {"modality": modality, "file": file,
                    "start": float(locator.rstrip("s"))}
    elif suffix == ".md":
        citation = {"modality": "field_note", "file": file, "location": locator}
    else:
        citation = {"modality": "occurrence_summary", "file": file, "field": locator}

    if text:
        citation["text"] = text
    return citation


def _dedupe(sources: list[dict]) -> list[dict]:
    """One citation per passage. Two claims about the same thing routinely cite
    the same line - and an inferred fact cites both of its inputs, which for
    "total minus APP" is one sentence naming both figures."""
    seen, unique = set(), []
    for source in sources:
        key = json.dumps(source, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique


def _claim(key, type_, value, sources, note=None, repeatable=False) -> dict:
    """One extractor's reading of one thing.

    repeatable marks a key that legitimately holds several values at once - a
    team has four members, a site has two machines. Without it _merge reads the
    second value as a source disagreeing with the first and reports a conflict,
    which for a list is never what happened.
    """
    return {"key": key, "type": type_, "value": value, "sources": sources,
            "note": note, "repeatable": repeatable}


def _comparable(value, type_: str | None = None):
    """Normalized form used only to decide whether two readings agree.

    Timestamps are compared to the minute. The Sumaúma app records a photo as
    08:31 while the stamp burned into that same photo reads 08:31:52 - the
    app truncates, the camera does not (the data README says the same of the
    coordinates it rounds). Treating that as two sources contradicting each
    other would fill the report with conflicts that are really one instant
    written at two precisions; the more precise reading is the one kept.
    """
    if type_ == "timestamp" and isinstance(value, str):
        return value[:16]                     # YYYY-MM-DDTHH:MM

    if isinstance(value, str):
        cleaned = value.strip().lower().replace(",", ".")
        cleaned = re.sub(r"[\s\-]+", "", cleaned)
        try:
            return round(float(cleaned), 4)
        except ValueError:
            return cleaned
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    if isinstance(value, (list, tuple)):
        return tuple(_comparable(v) for v in value)
    return value


def _flagged_segments(folder: Path) -> set[tuple[str, float]]:
    """(file, start) of every transcript segment Whisper itself distrusted."""
    for name in ("transcripts_standard.json", "transcripts_lite.json"):
        path = folder / ".cache" / name
        if path.exists():
            cache = json.loads(path.read_text(encoding="utf-8"))
            return {
                (audio, seg["start"])
                for audio, entry in cache.items()
                for seg in entry["segments"] if seg["flags"]
            }
    return set()


def _score(sources: list[dict], flagged: set[tuple[str, float]]) -> float:
    """Confidence follows from the evidence; it is never taken from a model.

    The best single source sets the ceiling, each independent corroborating
    source adds a little, and citing a segment Whisper flagged takes some
    back. See the fidelity caveat in the module docstring of the report
    drafter: a source is currently credited for backing a value if it was
    cited for it, not because the value was checked to occur in its text.
    """
    if not sources:
        return 0.0

    base = max(MODALITY_CONFIDENCE.get(s["modality"], 0.5) for s in sources)
    distinct = {(s["modality"], s["file"]) for s in sources}
    score = base + CORROBORATION_BONUS * (len(distinct) - 1)

    if any((s["file"], s.get("start")) in flagged for s in sources):
        score -= FLAGGED_SEGMENT_PENALTY

    return round(max(0.0, min(_MAX_CONFIDENCE, score)), 3)


# --------------------------------------------------------------------------
# one loader per cache file; each returns plain claims, merged further down
# --------------------------------------------------------------------------

def _photo_file(folder: Path, number: int) -> str:
    """The photo file the app's entry number refers to.

    Looked up by numeric stem rather than assumed to be <n>.jpg, so that the
    app's record and the stamp read off the pixels end up under the same fact
    key and can corroborate each other - which they cannot if one says
    "03.jpg" and the file on disk is "03.png".
    """
    directory = folder / "photos"
    if directory.is_dir():
        match = sorted(p.name for p in directory.glob(f"{number:02d}.*") if p.is_file())
        if match:
            return match[0]
    return f"{number:02d}.jpg"


def _from_occurrence_summary(folder: Path) -> list[dict]:
    path = folder / ".cache" / "occurrence_summary.json"
    if not path.exists():
        log.warning("no occurrence_summary.json - run extract_from_occurrence_summary.py")
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    claims = []

    simple_types = {
        "title": "location", "category": "classification",
        "opened_at": "timestamp", "car_registry": "registry",
        "area_estimate_ha": "area",
    }
    for key, type_ in simple_types.items():
        entry = data.get(key)
        if entry and entry.get("value") is not None:
            claims.append(_claim(key, type_, entry["value"],
                                 [_parse_source(entry["source"], entry.get("raw"))]))

    src = [{"modality": "occurrence_summary", "file": "occurrence-summary.txt",
            "field": "PONTOS MARCADOS"}]
    if data.get("marked_points"):
        claims.append(_claim("perimeter_points", "coordinate_set",
                             [p["coordinate"] for p in data["marked_points"]], src,
                             note=f"{len(data['marked_points'])} GPS fixes walked on foot"))

    for doc in data.get("documents", []):
        claims.append(_claim("auto_constatacao_number", "document", doc["number"],
                             [{"modality": "occurrence_summary",
                               "file": "occurrence-summary.txt",
                               "field": "DOCUMENTOS LAVRADOS",
                               "text": f"{doc['name']} {doc['number']}"}]))

    # The app's own per-photo record: independent of the pixels, so it can
    # corroborate (or contradict) what the burned-in stamp says.
    for index, photo in enumerate(data.get("photos", []), start=1):
        number = photo.get("n") or index
        name = _photo_file(folder, number)
        cite = [{"modality": "occurrence_summary", "file": "occurrence-summary.txt",
                 "field": f"FOTOS/{number:02d}", "text": photo.get("caption")}]
        if photo.get("timestamp"):
            claims.append(_claim(f"photo_{name}_timestamp", "timestamp", photo["timestamp"], cite))
        if photo.get("coordinate"):
            claims.append(_claim(f"photo_{name}_coordinate", "coordinate", photo["coordinate"], cite))
        if photo.get("caption"):
            claims.append(_claim(f"photo_{name}_caption", "description", photo["caption"], cite))

    return claims


def _from_photo_stamps(folder: Path) -> list[dict]:
    for name in ("photo_stamps_standard.json", "photo_stamps_lite.json"):
        path = folder / ".cache" / name
        if path.exists():
            break
    else:
        log.warning("no photo_stamps_*.json - run extract_text_from_photos.py")
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    types = {"timestamp": "timestamp", "coordinate": "coordinate", "agency": "organisation"}
    claims = []

    for photo, fields in data.items():
        for field, type_ in types.items():
            entry = fields.get(field)
            if not entry or entry.get("value") is None:
                continue
            claim = _claim(f"photo_{photo}_{field}", type_, entry["value"],
                           [_parse_source(entry["source"], entry.get("raw"))])
            # An extractor that already saw two readings disagree says so;
            # that disagreement is evidence in itself and is not dropped here.
            if entry.get("conflict"):
                claim["note"] = f"extractor flagged exif/stamp conflict: {entry['conflict']}"
            claims.append(claim)

    return claims


def _from_field_notes(folder: Path) -> list[dict]:
    path = folder / ".cache" / "field_notes.json"
    if not path.exists():
        log.warning("no field_notes.json - run extract_from_field_notes.py")
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    types = {
        "inspector_name": "person", "owner_name": "person", "operator_name": "person",
        "car_registry": "registry", "property_area_ha": "area",
        "reserve_legal_area_ha": "area", "total_suppressed_area_ha": "area",
        "app_area_ha": "area", "burned_area_estimate_ha": "area",
        "watercourse_name": "location", "auto_constatacao_number": "document",
        "legal_citation": "legal_basis", "signature_refused": "procedural",
    }
    claims = []

    def sources_of(entry):
        """One citation per passage, each with the source it really came from.

        entry["citations"] pairs them explicitly. Before it existed this split
        entry["source"] and entry["raw"] and zipped them by position - but the
        ids were a sorted set and the quotes an unsorted list, so every field
        note was filed under another block's timestamp. A cache without
        citations is therefore cited by file only: the pairing in it is not
        recoverable, and guessing it is how the wrong quote gets printed under
        a value.
        """
        citations = entry.get("citations")
        if citations is not None:
            return [_parse_source(c["source"], c.get("quote")) for c in citations]

        log.warning("%s has no citations - cache predates them, quoting nothing; "
                    "re-run extract_from_field_notes.py", entry.get("value"))
        return [_parse_source(part) for part in (entry.get("source") or "").split(" + ") if part]

    for key, entry in data.items():
        if key == "equipment":
            for item in entry:
                if item.get("value"):
                    claims.append(_claim(f"equipment_{item.get('type', 'outro')}",
                                         "equipment", item["value"], sources_of(item),
                                         repeatable=True))
        elif key == "team_members":
            for member in entry:
                if member.get("value"):
                    claims.append(_claim("team_member", "person", member["value"],
                                         sources_of(member), repeatable=True))
        elif isinstance(entry, dict) and entry.get("value") is not None:
            claims.append(_claim(key, types.get(key, "other"), entry["value"], sources_of(entry)))

    return claims


# --------------------------------------------------------------------------


def _merge(claims: list[dict], flagged: set) -> list[dict]:
    """Fold claims about the same key into one fact, preserving disagreement."""
    facts = []

    by_key: dict[str, list[dict]] = {}
    for claim in claims:
        by_key.setdefault(claim["key"], []).append(claim)

    for key, group in by_key.items():
        type_ = group[0]["type"]
        readings: dict = {}
        for claim in group:
            readings.setdefault(_comparable(claim["value"], type_), []).append(claim)

        sources = _dedupe([s for claim in group for s in claim["sources"]])
        notes = [c["note"] for c in group if c.get("note")]

        if any(c.get("repeatable") for c in group):
            # Several values under one key because there are several of the
            # thing, not because two sources disagree about one of them.
            for reading in readings.values():
                reading_sources = _dedupe([s for c in reading for s in c["sources"]])
                facts.append({
                    "key": key,
                    "type": type_,
                    "value": reading[0]["value"],
                    "status": "observed",
                    "confidence": _score(reading_sources, flagged),
                    "sources": reading_sources,
                })
            continue

        if len(readings) == 1:
            # Readings that agree can still differ in precision. A truncated
            # time is padded back to ":00" seconds, so it is the same length
            # as a real one - prefer whichever actually carries seconds.
            value = group[0]["value"]
            if type_ == "timestamp":
                value = next((c["value"] for c in group
                              if not str(c["value"]).endswith(":00")), value)
            fact = {
                "key": key,
                "type": type_,
                "value": value,
                "status": "observed",
                "confidence": _score(sources, flagged),
                "sources": sources,
            }
        else:
            # Two modalities read the same thing differently. Neither is
            # discarded: the report drafter needs to show a human both.
            variants = []
            for reading in readings.values():
                reading_sources = _dedupe([s for c in reading for s in c["sources"]])
                variants.append({
                    "value": reading[0]["value"],
                    "sources": reading_sources,
                    "confidence": _score(reading_sources, flagged),
                })
            best = max(variants, key=lambda v: v["confidence"])
            fact = {
                "key": key,
                "type": group[0]["type"],
                "value": best["value"],
                "status": "conflicting",
                "confidence": round(best["confidence"] * 0.9, 3),
                "variants": variants,
                "sources": sources,
            }
            log.warning("%s: %d readings disagree -> %s", key, len(readings),
                        [v["value"] for v in variants])

        if notes:
            fact["note"] = "; ".join(notes)
        facts.append(fact)

    return facts


def _infer(facts: list[dict]) -> list[dict]:
    """Values the report states that no source states, computed from ones it does.

    Only arithmetic with a shown derivation belongs here - never a guess
    dressed up as a calculation.
    """
    by_key = {f["key"]: f for f in facts if f["status"] != "missing"}
    inferred = []

    total, app = by_key.get("total_suppressed_area_ha"), by_key.get("app_area_ha")
    if total and app:
        try:
            remainder = round(float(str(total["value"]).replace(",", ".")) -
                              float(str(app["value"]).replace(",", ".")), 4)
        except ValueError:
            return inferred
        inferred.append({
            "key": "reserve_legal_suppressed_ha",
            "type": "area",
            "value": remainder,
            "status": "inferred",
            "confidence": round(min(total["confidence"], app["confidence"]) * 0.95, 3),
            "derivation": (f"total_suppressed_area_ha ({total['value']}) - "
                           f"app_area_ha ({app['value']}) = {remainder}"),
            "sources": _dedupe(total["sources"] + app["sources"]),
        })

    return inferred


def build(folder: Path) -> dict:
    flagged = _flagged_segments(folder)
    if flagged:
        log.info("%d transcript segment(s) flagged by whisper, cited facts penalised", len(flagged))

    claims = (_from_occurrence_summary(folder) + _from_photo_stamps(folder)
              + _from_field_notes(folder))
    log.info("%d raw claims from %d extractor file(s)", len(claims), 3)

    facts = _merge(claims, flagged)
    facts += _infer(facts)

    for key, (reason, reason_en) in _KNOWN_ABSENT.items():
        if key not in {f["key"] for f in facts}:
            facts.append({"key": key, "type": "other", "value": None,
                          "status": "missing", "confidence": 0.0,
                          "reason": reason, "reason_en": reason_en, "sources": []})

    facts.sort(key=lambda f: (f["status"] != "conflicting", -f["confidence"], f["key"]))
    for index, fact in enumerate(facts, start=1):
        fact["id"] = f"fact_{index:03d}"

    counts: dict[str, int] = {}
    for fact in facts:
        counts[fact["status"]] = counts.get(fact["status"], 0) + 1

    return {"case": folder.name, "facts": facts, "summary": counts}


def main(folder: Path) -> None:
    folder = Path(folder)
    if not (folder / ".cache").is_dir():
        log.error("no .cache/ inside %s - run the extractors first", folder)
        raise SystemExit(1)

    evidence = build(folder)

    # .cache/ holds each extractor's regenerable intermediate; output/ holds
    # what the pipeline is actually for, and what a person opens.
    out_path = folder / "output" / "evidence.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("%d facts: %s", len(evidence["facts"]),
             ", ".join(f"{n} {s}" for s, n in sorted(evidence["summary"].items())))
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("folder", nargs="?", default=".")
    args = parser.parse_args()

    main(Path(args.folder))
