"""
Extract entities from the two free-text sources - field-notes.md (typed
one-handed, typos and abbreviations intact) and the audio transcripts
(dictated, and mangled wherever Whisper misheard a name: "Cleidiane" comes
back "Clay de Ano", "Deuzimar" comes back "Deus e Mar"). Neither source alone
is reliable for names; read together, one's clean spelling and the other's
extra detail corroborate each other.

A model generating text can invent a value that is nowhere in the input, so
this extractor carries its own safety net: the model must return, for every
fact, the exact source passage it read it from, and every passage it returns
has to pass two checks after the call - it must exist verbatim in some source,
and it must be about the value it is offered for; see _verify. Nothing is
trusted on the model's word, including which source it claims to have read.

Fields are asked for in five small groups rather than all at once. One call for
everything is how a 4b model ends up answering with the property's area and
never mentioning the area actually suppressed, which is the figure the report
exists to state; see _GROUPS. A fact quoted from two different files is
reported at higher confidence than one quoted from a single file, the same
shape as the EXIF/stamp agreement bonus in extract_text_from_photos.py.

Runs against the local Qwen3-VL weights via Ollama - see
download_vision_language_models.py. No cloud call, no per-document cost.

Usage:
    ollama pull qwen3-vl:4b                    # once, via download_vision_language_models.py
    python extract_from_field_notes.py data/altamira
"""

import argparse
import json
import re
from pathlib import Path

import ollama

from facts import (fact, missing, normalize, strip_reasoning,
                   value_supported_by)
from logconf import setup

log = setup("extract_field_notes")

MODEL_TAG = "qwen3-vl:4b"

_NOTE_RE = re.compile(r"\*\*(\d{2}:\d{2})\*\*\s*\n```\n(.*?)\n```", re.S)

# One grounded claim: the value, and one passage the model read it in.
#
# It is deliberately not asked for a list of citations, nor for which source
# each came from. The source id was never trusted anyway - _find_source looks
# the passage up - and asking a 4b model to enumerate every place a value
# appears sent it into a repetition loop that truncated the JSON and lost the
# whole group. One quote is all that is needed as proof it read the value
# somewhere; _citations then finds every other passage that supports the same
# value by searching the sources, which is exhaustive and verified by
# construction rather than by the model's diligence.
_FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "string"},
        "quote": {"type": "string"},
    },
    "required": ["value", "quote"],
}

_EQUIPMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "string"},          # brand and model only: "Komatsu D51"
        "type": {"type": "string", "enum": ["trator", "motosserra", "outro"]},
        "brand": {"type": "string"},
        "model": {"type": "string"},
        "quote": _FACT_SCHEMA["properties"]["quote"],
    },
    "required": ["value", "type", "quote"],
}

# Fields are asked for in small related groups, one call each, rather than all
# of them in one call. A 4b model handed fifteen fields and fifty passages at
# once answers the two or three it noticed and drops the rest in silence: it
# returned the property's 412,5 ha and missed the 23,418 ha actually suppressed,
# which is the number the report is about. Each group also carries a one-line
# description per field, because the distinction between four different area
# figures in the same day's notes is the whole difficulty here.
_GROUPS = {
    "pessoas": {
        "inspector_name": "quem narra as notas, o fiscal responsável",
        "owner_name": "quem se identificou como proprietário ou responsável pelo imóvel",
        "operator_name": "quem operava o equipamento, ou se declarou tomador de conta",
        "team_members": "demais integrantes da equipe de fiscalização citados pelo nome",
    },
    "areas": {
        "total_suppressed_area_ha": "área total suprimida/derrubada apurada, em hectares",
        "app_area_ha": ("parte da área suprimida que está em Área de Preservação "
                        "Permanente (APP), em hectares"),
        "property_area_ha": "área total do imóvel registrada no CAR, em hectares",
        "reserve_legal_area_ha": "área de Reserva Legal do imóvel, em hectares",
        "burned_area_estimate_ha": "estimativa da área queimada, em hectares",
    },
    "local": {
        "watercourse_name": "nome do igarapé, rio ou curso d'água afetado",
        "car_registry": "número de registro no CAR do imóvel",
    },
    "documentos": {
        "auto_constatacao_number": "número do Auto de Constatação, no formato NNNNN/AAAA",
        "legal_citation": "dispositivo legal citado como enquadramento da infração",
        "signature_refused": "responda 'sim' apenas se o autuado se recusou a assinar",
    },
    "equipamentos": {
        "equipment": ("cada máquina encontrada no local. Em 'value' apenas marca e "
                      "modelo, sem a palavra trator/motosserra, que vai em 'type'"),
    },
}

# Fields whose value is a reading of what the passage means rather than a value
# copied out of it: a note saying the autuado "recusou assinar" supports
# signature_refused="sim" without containing the word, so the passage-mentions-
# the-value test in _verify does not apply to them.
_INFERRED_FIELDS = {"signature_refused"}

_LIST_FIELDS = {"team_members", "equipment"}

# A day's work yields a handful of machines and team members, not dozens. The
# cap is there because an unbounded array is what the model loops on.
_MAX_LIST_ITEMS = 6


def _group_schema(fields: dict[str, str]) -> dict:
    """The JSON schema for one group: every field optional, so a field with no
    evidence in this case comes back absent instead of guessed."""
    properties = {}
    for name in fields:
        item = _EQUIPMENT_SCHEMA if name == "equipment" else _FACT_SCHEMA
        properties[name] = ({"type": "array", "items": item, "maxItems": _MAX_LIST_ITEMS}
                            if name in _LIST_FIELDS else item)
    return {"type": "object", "properties": properties, "required": []}


_SYSTEM_PROMPT = """\
You extract facts for a Brazilian environmental enforcement report from field
notes and voice-note transcripts written by the inspector on site.

Rules, no exceptions:
1. Every value you return must be copied or directly inferable from the
   provided source text - never invented, never filled in from what would be
   typical or expected.
2. Every fact must quote, in "quote", the one passage you read the value in,
   verbatim and complete, exactly as it appears in that source (same spelling,
   same typos, same case) - do not correct, translate or clean it up. Quote the
   passage that states the value itself, never a nearby one. One passage is
   enough; do not repeat yourself.
3. If a field is not stated anywhere in the sources, omit it entirely. Do not
   guess, and do not use null as a placeholder for a real value.
4. Copy numbers exactly as written, with the same decimal separator and the
   same number of digits: 23,418 stays 23,418 and never becomes 23,42 or 23.
   Several different area figures appear in one day of notes - the property,
   its legal reserve, what was cleared, what burned. Return each under the
   field that asks for it, and if you cannot tell which one a figure is,
   omit it.
5. Audio transcripts contain speech-to-text errors, especially in names
   ("Clay de Ano" is almost certainly "Cleidiane" said aloud). You may
   normalize a value to its most likely correct form, but the quote you cite
   must still be the literal mangled source text you read it from.
6. Write proper nouns in standard Portuguese form, with capitals and accents:
   a creek typed "igarape santa rosa" with no accents is "Igarapé Santa Rosa".
   This is a legal document and it has to read as one.
"""


def _load_field_notes(folder: Path) -> dict[str, str]:
    """One source per timestamped block, id'd as 'field-notes.md@HH:MM'."""
    text = (folder / "field-notes.md").read_text(encoding="utf-8")
    return {
        f"field-notes.md@{stamp}": body.strip()
        for stamp, body in _NOTE_RE.findall(text)
    }


def _load_transcripts(folder: Path) -> dict[str, str]:
    """One source per Whisper segment, id'd as '<file>@<start>s'."""
    for name in ("transcripts_standard.json", "transcripts_lite.json"):
        cache_file = folder / ".cache" / name
        if cache_file.exists():
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
            log.info("using %s for transcript text", name)
            break
    else:
        log.error("no transcripts_standard.json or transcripts_lite.json in %s/.cache "
                   "- run extract_transcriptions.py first", folder)
        raise SystemExit(1)

    return {
        f"{audio}@{seg['start']}s": seg["text"]
        for audio, entry in cache.items()
        for seg in entry["segments"]
    }


def _build_prompt(sources: dict[str, str]) -> str:
    lines = [f"[{source_id}] {text}" for source_id, text in sources.items()]
    return "SOURCES:\n" + "\n".join(lines)


def _find_source(quote: str, sources: dict[str, str]) -> str | None:
    """Which source actually contains this quote, if any.

    The model's own claim about which source it read a passage from is not
    taken at face value: qwen3-vl routinely cites a field-notes.md block for a
    line that is really in an audio transcript. The passage itself is the
    evidence, so it is looked up across every source and attributed to wherever
    it genuinely appears. A passage found nowhere is not evidence at all and its
    fact is dropped. Accents and case are ignored in the lookup - a small model
    dropping an accent off a quote it otherwise copied exactly should not cost
    a real citation.
    """
    needle = normalize(quote)
    if not needle:
        return None
    return next((sid for sid, text in sources.items() if needle in normalize(text)), None)


def _narrowest(text: str, value) -> str:
    """The smallest part of a source that still states the value.

    A field-note block is several lines typed at one timestamp; quoting the
    whole block for a number written on one of them buries the citation in
    unrelated text. Transcript segments are single lines already and come back
    unchanged.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return text.strip()
    return next((line for line in lines if value_supported_by(line, value)), text.strip())


def _citations(value, quote: str, sources: dict[str, str], field: str) -> list[dict]:
    """Every passage in the sources that states this value.

    The model proposes; this verifies. A value survives if and only if some
    real passage states it, and the citations are exactly those passages -
    found by searching every source, not by asking the model which ones agree.
    That makes corroboration exhaustive and every citation about the value by
    construction: it caught 412,5 ha being stated twice, typed in the notes and
    dictated into 08.mp3, which the model never mentioned.

    The model's own quote is checked too, but only as a signal about the model:
    it does not get a veto. It quoted "se identificou como Valdenir" for the
    value "Valdenir Moraes Cardoso" - a real passage, and the wrong one, while
    the notes state the full name two lines away. Discarding a value the
    sources do state, because the model pointed at the wrong line, loses a fact
    for no gain in safety.

    Fields in _INFERRED_FIELDS have no literal value to search for, so for them
    the quoted passage is the only citation there can be, and it must exist.
    """
    quote = (quote or "").strip()
    anchor = _find_source(quote, sources)
    if anchor is None:
        log.warning("%s=%r: quoted passage is in no source: %r", field, value, quote)

    if field in _INFERRED_FIELDS:
        return [{"source": anchor, "quote": quote}] if anchor else []

    if anchor and not value_supported_by(quote, value):
        log.info("%s=%r: model quoted a passage that does not state it (%s) - "
                 "searching the sources instead", field, value, anchor)

    return [{"source": sid, "quote": _narrowest(text, value)}
            for sid, text in sources.items() if value_supported_by(text, value)]


def _verify(value_obj: dict, sources: dict[str, str], field: str) -> dict:
    """One model answer -> one fact, or nothing if it cannot be backed.

    Confidence reflects how many distinct *files* back the value: two passages
    on one page of notes are one source saying it twice, while the same figure
    typed in the notes and dictated into a voice note is two. It is a rough
    figure for whoever reads this cache - build_evidence recomputes confidence
    from the modality of each citation, and that one is authoritative.
    """
    value = value_obj.get("value")
    if not value:
        return missing()

    citations = _citations(value, value_obj.get("quote"), sources, field)
    if not citations:
        return missing()

    files = {c["source"].split("@")[0] for c in citations}
    confidence = 0.82 if len(files) > 1 else 0.68
    return fact(value, " + ".join(c["source"] for c in citations), confidence,
                "; ".join(c["quote"] for c in citations), citations=citations)

def _ask(model: str, sources_block: str, group: str, fields: dict[str, str]) -> dict:
    """One call for one group of fields. Raw model output, nothing trusted yet.

    The system prompt and the sources block are byte-identical on every call so
    that ollama can reuse the prompt's KV cache across groups: only the short
    question at the end changes, which is what makes five focused calls cost
    little more than one unfocused one.
    """
    asked = "\n".join(f"- {name}: {description}" for name, description in fields.items())
    response = ollama.chat(
        model=model,
        format=_group_schema(fields),
        think=False,
        options={"temperature": 0},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": sources_block},
            {"role": "user", "content": f"Extraia somente estes campos:\n{asked}\n\n"
                                        "Omita qualquer campo que as fontes não afirmem."},
        ],
    )

    content = strip_reasoning(response.message.content or response.message.thinking)
    if not content:
        log.error("%s: model returned nothing", group)
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        log.error("%s: model output was not valid JSON, group skipped:\n%s", group, content)
        return {}


def extract(folder: Path, model: str = MODEL_TAG) -> dict:
    """Every field the two free-text sources support, each with its citations."""
    sources = _load_field_notes(folder) | _load_transcripts(folder)
    log.info("%d source passages (%d field-note blocks, %d transcript segments)",
             len(sources),
             sum(1 for sid in sources if sid.startswith("field-notes.md")),
             sum(1 for sid in sources if not sid.startswith("field-notes.md")))

    sources_block = _build_prompt(sources)
    result: dict = {}

    for group, fields in _GROUPS.items():
        raw = _ask(model, sources_block, group, fields)
        found = []

        for name in fields:
            offered = raw.get(name)

            if name not in _LIST_FIELDS:
                verified = _verify(offered, sources, name) if offered else missing()
                result[name] = verified
                if verified["value"] is not None:
                    found.append(f"{name}={verified['value']!r}")
                continue

            verified_items = []
            for item in offered or []:
                entry = _verify(item, sources, name)
                if entry["value"] is None:
                    continue
                if name == "equipment":
                    entry |= {key: item.get(key) for key in ("type", "brand", "model")}
                verified_items.append(entry)
            result[name] = verified_items
            found += [f"{name}={item['value']!r}" for item in verified_items]

        log.info("%s: %s", group, ", ".join(found) or "nothing supported by the sources")

    return result


def main(folder: Path, model: str = MODEL_TAG) -> None:
    folder = Path(folder)
    if not (folder / "field-notes.md").is_file():
        log.error("no field-notes.md inside %s", folder)
        raise SystemExit(1)

    result = extract(folder, model)

    out_path = folder / ".cache" / "field_notes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    dropped = [k for k, v in result.items() if isinstance(v, dict) and v["value"] is None]
    if dropped:
        log.warning("%d field(s) with no verifiable evidence: %s", len(dropped), ", ".join(dropped))
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("folder", nargs="?", default=".")
    parser.add_argument("--model", default=MODEL_TAG, help="Ollama model tag")
    args = parser.parse_args()

    main(Path(args.folder), model=args.model)
