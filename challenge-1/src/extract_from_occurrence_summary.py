"""
Parse occurrence-summary.txt, the Sumaúma app's fixed-format export - not a
model job. The app writes the same section order and layout every time, so a
line parser is both cheaper and more reliable here than any model would be:
there is no ambiguity in the input for a model's flexibility to help with.

Six sections, always in this order, separated by a blank line:

    SUMAÚMA - FICHA DA OCORRÊNCIA        <- title line, not a field
    Título: ...                          <- key: value header block
    PONTOS MARCADOS (n)                  <- timestamped GPS fixes
    NOTA ESCRITA                         <- free text, duplicates field-notes.md
    FOTOS (n)                            <- one fix + one caption line per photo
    ÁUDIOS (n)                           <- one fix per voice note, no caption
    DOCUMENTOS LAVRADOS (n página(s))    <- issued document name + number

NOTA ESCRITA is kept as one raw block and never parsed for entities here: it
is the same free text field-notes.md carries, already destined for the
field-notes extractor, and parsing it twice would just be two chances to
disagree with itself.

Usage:
    python extract_from_occurrence_summary.py data/altamira
"""

import argparse
import json
import re
from pathlib import Path

from extract_text_from_photos import parse_coordinate_pair, parse_stamp_timestamp
from facts import fact, missing
from logconf import setup

log = setup("extract_occurrence_summary")

_HEADER_FIELDS = {
    "Título": "title",
    "Categoria": "category",
    "Aberta em": "opened_at",
    "Registro CAR": "car_registry",
    "Área estimada": "area_estimate_ha",
}

_SECTION_RE = re.compile(r"^([A-ZÀ-Ú ]+?)(?:\s*\((\d+)[^)]*\))?\s*$")

_FIX_RE = re.compile(
    r"(?P<date>\d{2}/\d{2}/\d{4})\s+(?P<time>\d{2}:\d{2})\s+"
    r"(?P<coord>[\d°'\".NSEOWL ]+?)\s*\(±(?P<accuracy>\d+)\s*m\)"
)

_NUMBERED_RE = re.compile(r"^(?P<n>\d+)\.\s*(?P<rest>.*)$")

_DOCUMENT_RE = re.compile(
    r"^(?P<name>.+?)\s+(?P<number>\d+/\d{4})\s+p[aá]gina\s+(?P<page>\d+)$", re.I
)


def _parse_ha(text: str) -> float | None:
    """'23,4 ha' -> 23.4 - Brazilian decimal comma, unit stripped."""
    match = re.search(r"([\d.,]+)\s*ha", text)
    return float(match.group(1).replace(".", "").replace(",", ".")) if match else None


def _parse_fix(line: str) -> dict | None:
    """One 'date time coord (±accuracy m)' reading -> timestamp/coordinate/accuracy."""
    match = _FIX_RE.search(line)
    if not match:
        return None

    return {
        "timestamp": parse_stamp_timestamp(f"{match['date']} {match['time']}"),
        "coordinate": parse_coordinate_pair(match["coord"]),
        "accuracy_m": int(match["accuracy"]),
    }


def _split_sections(text: str) -> list[str]:
    return [block.strip("\n") for block in text.strip().split("\n\n") if block.strip()]


def _parse_header(block: str) -> dict:
    fields = {}
    for line in block.splitlines():
        key, _, value = line.partition(":")
        name = _HEADER_FIELDS.get(key.strip())
        if name is None:
            continue
        value = value.strip()
        fields[name] = _parse_ha(value) if name == "area_estimate_ha" else value
    return fields


def _parse_points(lines: list[str]) -> list[dict]:
    return [fix for line in lines if (fix := _parse_fix(line)) is not None]


def _parse_captioned_fixes(lines: list[str]) -> list[dict]:
    """FOTOS' shape: a numbered fix line, then one indented caption line.

    The app's own number is kept: it is what binds an entry to photos/03.jpg.
    Position in this list is not the same thing - one unparseable fix line and
    every later photo would be off by one, quietly filing photo 4's coordinates
    under photo 3.
    """
    entries = []
    pending = None

    for line in lines:
        numbered = _NUMBERED_RE.match(line.strip())
        if numbered:
            pending = _parse_fix(numbered["rest"])
            if pending is None:
                log.warning("fix line not understood, entry skipped: %r", line)
                continue
            pending["n"] = int(numbered["n"])
            pending["caption"] = None
            entries.append(pending)
        elif pending is not None:
            pending["caption"] = line.strip()

    return entries


def _parse_documents(lines: list[str]) -> list[dict]:
    documents = []
    for line in lines:
        numbered = _NUMBERED_RE.match(line.strip())
        if not numbered:
            continue
        match = _DOCUMENT_RE.match(numbered["rest"])
        if match is None:
            log.warning("documento lavrado not understood: %r", line)
            continue
        documents.append({
            "name": match["name"].strip(),
            "number": match["number"],
            "page": int(match["page"]),
        })
    return documents


def parse_occurrence_summary(path: Path) -> dict:
    """Parse one occurrence-summary.txt into its six sections."""
    sections = _split_sections(Path(path).read_text(encoding="utf-8"))

    header: dict = {}
    parsed = {
        "marked_points": [],
        "written_note": None,
        "photos": [],
        "audios": [],
        "documents": [],
    }

    for block in sections:
        heading, *rest = block.splitlines()
        match = _SECTION_RE.match(heading)

        if match is None:
            header.update(_parse_header(block))          # title line, or a stray header block
            continue

        name = match[1].strip()
        if name == "PONTOS MARCADOS":
            parsed["marked_points"] = _parse_points(rest)
        elif name == "NOTA ESCRITA":
            parsed["written_note"] = "\n".join(rest).strip() or None
        elif name == "FOTOS":
            parsed["photos"] = _parse_captioned_fixes(rest)
        elif name == "ÁUDIOS":
            parsed["audios"] = _parse_captioned_fixes(rest)
        elif name == "DOCUMENTOS LAVRADOS":
            parsed["documents"] = _parse_documents(rest)
        else:
            header.update(_parse_header(block))

    # Cited by name, relative to the case folder: the same file must not read
    # as "occurrence-summary.txt" from one working directory and
    # "../data/altamira/occurrence-summary.txt" from another.
    rel = Path(path).name

    return {
        "title": fact(header.get("title"), f"{rel}#Título", 0.99, header.get("title")),
        "category": fact(header.get("category"), f"{rel}#Categoria", 0.99, header.get("category")),
        "opened_at": (
            fact(parse_stamp_timestamp(header["opened_at"]), f"{rel}#Aberta em", 0.99, header["opened_at"])
            if header.get("opened_at") else missing()
        ),
        "car_registry": fact(header.get("car_registry"), f"{rel}#Registro CAR", 0.99, header.get("car_registry")),
        "area_estimate_ha": (
            fact(header["area_estimate_ha"], f"{rel}#Área estimada", 0.99, header["area_estimate_ha"])
            if header.get("area_estimate_ha") is not None else missing()
        ),
        **parsed,
    }


def main(folder: Path) -> None:
    folder = Path(folder)
    source = folder / "occurrence-summary.txt"
    if not source.is_file():
        log.error("no occurrence-summary.txt inside %s", folder)
        raise SystemExit(1)

    result = parse_occurrence_summary(source)

    out_path = folder / ".cache" / "occurrence_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("%s: %d marked points, %d photos, %d audios, %d documents",
             folder.name, len(result["marked_points"]), len(result["photos"]),
             len(result["audios"]), len(result["documents"]))
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("folder", nargs="?", default=".")
    args = parser.parse_args()

    main(Path(args.folder))
