"""
Turn final_report.jpg into two files: the shape of the report, and what this
particular one says.

    data/<case>/report_template.json    empty slots + where each is filled from
    data/<case>/report_reference.json   the values read off the photograph

They are deliberately separate, because final_report.jpg is the answer. The
template is the form: the eight numbered sections a Relatório de Fiscalização
always has, which are printed boilerplate and identical across municipalities.
The reference is this case's filled-in values, and it is QUARANTINED - the
drafting step must never read it. A draft built from the reference would be a
copy of the target, not a report assembled from field evidence, and it would
tell you nothing about whether the pipeline works. Its purpose is scoring:
diff the evidence-built draft against it and see how much of a real report
the evidence can actually support.

Only the values are read by the model. The slot structure is written out by
hand in generate_report.REPORT_SLOTS - shared with the drafter so the two can
never disagree about the form, and kept out of the model's reach so that a VLM
misreading a heading cannot corrupt the template every later step depends on.

Labels stay in Portuguese: this is a Brazilian legal instrument and the
filed report has to read as one. The English label alongside each is what the
UI's English view shows; the Portuguese wording is the one of record.

Usage:
    python extract_from_final_report.py data/altamira
    python extract_from_final_report.py data/altamira --template-only   # no model needed
"""

import argparse
import json
import re
from pathlib import Path

import ollama

from facts import strip_reasoning
from generate_report import REPORT_SLOTS
from logconf import setup

log = setup("extract_final_report")

MODEL_TAG = "qwen3-vl:4b"

# One transcription at a time, so the model has one thing to do per call.
#
# Asking for all eight sections in one schema-constrained call is how this
# quietly returned half the page: with `required` empty, the grammar lets the
# model close the JSON whenever it likes, and qwen3-vl:4b answered the four
# sections it latched onto and dropped Autor, Local, Objetivo and the legal
# basis - all four plainly legible on a clean 1054x1539 photograph. Per section,
# the field is required, so "I could not read it" has to be an explicit empty
# answer rather than a key that silently never appears.
_SECTION_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}

# Only where the printed heading is not enough to find the text on the page.
_SECTION_HINTS = {
    "enquadramento": ("o dispositivo legal citado ao final da página, em itálico "
                      "(lei, artigo e inciso). Não é uma seção numerada"),
}

_SYSTEM_PROMPT = """\
You transcribe one section of a Brazilian environmental inspection report
(Relatório de Fiscalização) from a photograph of the printed page.

Rules, no exceptions:
1. Copy what is printed in that section, verbatim and in Portuguese. Do not
   summarize, translate, correct spelling, or complete anything cut off.
2. Return the content only - not the section number, not its title.
3. If that section is not legible in the photograph, return an empty string.
   Never write what such a section usually says.
"""


def _strip_heading(text: str, slot: dict) -> str:
    """Drop the section's own number and title if the model repeated them.

    It is asked not to and does anyway ("2. Autor: Ivandro Sampaio Correia..."),
    which then prints the heading twice in any view that has a heading of its
    own. Removing it here is safe because only a prefix matching this section's
    known number and label is touched - the content itself is never rewritten.
    """
    pattern = r"^\s*"
    if slot["number"]:
        pattern += rf"(?:{slot['number']}\s*[.)-]\s*)?"
    pattern += rf"(?:{re.escape(slot['label'])}\s*:?\s*)?"
    stripped = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()
    if stripped != text.strip():
        log.debug("%s: stripped the repeated heading", slot["id"])
    return stripped or text.strip()


def _ask_for_section(image: Path, slot: dict, model: str) -> str:
    """Transcribe one section. Empty string means the model could not read it.

    The system prompt and the message carrying the image are byte-identical on
    every call, so ollama reuses the vision prefill across the eight sections
    instead of re-encoding the photograph each time.
    """
    heading = f"{slot['number']}. {slot['label']}" if slot["number"] else slot["label"]
    hint = _SECTION_HINTS.get(slot["id"])
    asked = f"Transcreva a seção “{heading}”" + (f" - {hint}" if hint else "") + "."

    response = ollama.chat(
        model=model,
        format=_SECTION_SCHEMA,
        think=False,
        options={"temperature": 0},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",
             "content": "Esta imagem é a página de um Relatório de Fiscalização.",
             "images": [str(image)]},
            {"role": "user", "content": asked},
        ],
    )

    content = strip_reasoning(response.message.content or response.message.thinking)
    if not content:
        log.warning("%s: model returned nothing", slot["id"])
        return ""
    try:
        return _strip_heading((json.loads(content).get("text") or "").strip(), slot)
    except json.JSONDecodeError:
        log.error("%s: model output was not valid JSON:\n%s", slot["id"], content)
        return ""


def build_template() -> dict:
    """The empty form: every slot, its label, and where the drafter fills it from."""
    return {
        "document": "Relatório de Fiscalização",
        "issuer": "SEMMA - Secretaria Municipal da Gestão do Meio Ambiente, Altamira/PA",
        "language": "pt-BR",
        "slots": [
            {
                "id": slot["id"],
                "number": slot["number"],
                "label": slot["label"],
                "label_en": slot["label_en"],
                "kind": slot["kind"],
                "value": slot.get("value") if slot["kind"] == "static" else None,
                "status": "static" if slot["kind"] == "static" else "empty",
                "fills_from": slot["fills_from"],
                **({"coordinate_rule": slot["coordinate_rule"]}
                   if "coordinate_rule" in slot else {}),
                **({"expected_missing": True} if slot.get("expected_missing") else {}),
                "fact_ids": [],
                "confidence": None,
            }
            for slot in REPORT_SLOTS
        ],
    }


def read_reference(image: Path, model: str = MODEL_TAG) -> dict:
    """Transcribe the photographed report, one section per call.

    Scoring only - never a pipeline input. A section the model cannot read is
    left out of "values" entirely, which is what compare_report reads as "not
    transcribed" and keeps distinct from a section that is blank on the paper.
    """
    slots = [slot for slot in REPORT_SLOTS if slot["kind"] != "static"]
    log.info("reading %s with %s, %d sections", image.name, model, len(slots))

    values = {}
    for slot in slots:
        text = _ask_for_section(image, slot, model)
        if text:
            values[slot["id"]] = text
            log.info("  %-14s %d chars", slot["id"], len(text))
        else:
            log.warning("  %-14s not legible to the model - left out", slot["id"])

    return {
        "source": str(image),
        "warning": ("QUARANTINED. Read off the answer document for scoring a draft "
                    "against it. Never use as an input to drafting."),
        "model": model,
        "values": values,
    }


def main(folder: Path, model: str = MODEL_TAG, template_only: bool = False) -> None:
    folder = Path(folder)

    template_path = folder / "report_template.json"
    template_path.write_text(json.dumps(build_template(), ensure_ascii=False, indent=2),
                             encoding="utf-8")
    log.info("wrote %s (%d slots)", template_path, len(REPORT_SLOTS))

    if template_only:
        return

    image = folder / "final_report.jpg"
    if not image.is_file():
        log.error("no final_report.jpg inside %s", folder)
        raise SystemExit(1)

    reference = read_reference(image, model)

    reference_path = folder / "report_reference.json"
    reference_path.write_text(json.dumps(reference, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    read = sorted(reference["values"])
    total = sum(1 for slot in REPORT_SLOTS if slot["kind"] != "static")
    log.info("read %d of %d sections from the photograph: %s",
             len(read), total, ", ".join(read))
    log.info("wrote %s", reference_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("folder", nargs="?", default=".")
    parser.add_argument("--model", default=MODEL_TAG, help="Ollama model tag")
    parser.add_argument("--template-only", action="store_true",
                        help="write the empty template and skip reading the photograph")
    args = parser.parse_args()

    main(Path(args.folder), model=args.model, template_only=args.template_only)
