"""
Compare the generated report (output/report.json) against the original,
municipality-filled report as read off the photograph (report_reference.json,
written by extract_from_final_report.py).

report_reference.json is QUARANTINED from drafting - generate_report.py never
reads it - but it is exactly the reference this module needs: the point of the
exercise is to see how much of a real report the field evidence alone can
support, and where it cannot.

Every difference is filed under one of seven verdicts, never folded into another:

    MATCH                  every value the generated report states is in the original
    PARTIAL_MATCH          some of them are, and the rest were not found in its text
    MISSING_FROM_GENERATED original states it, generated correctly declines
    EXTRA_IN_GENERATED     generated states it, its text does not appear in the
                           corresponding part of the original
    CONFLICT               both state a value, and the values disagree
    UNSUPPORTED            generated states something its own cited fact does
                           not back - a validator failure, surfaced not swallowed
    NOT_COMPARABLE         one side gives us nothing to judge the other by: the
                           original slot was never transcribed off the photo, or
                           the generated text makes no checkable point claim

A verdict is never "error": MISSING_FROM_GENERATED is frequently the correct,
honest outcome - the evidence really does not support the field - and the note
says so rather than implying the pipeline failed.

Notes are emitted as a code plus arguments and rendered in either language by
note_text(), so the UI and the CLI say the same thing in pt-BR and English.

Usage:
    python compare_report.py data/altamira
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from facts import normalize
from logconf import setup

log = setup("compare_report")

MATCH = "MATCH"
PARTIAL_MATCH = "PARTIAL_MATCH"
MISSING_FROM_GENERATED = "MISSING_FROM_GENERATED"
EXTRA_IN_GENERATED = "EXTRA_IN_GENERATED"
CONFLICT = "CONFLICT"
UNSUPPORTED = "UNSUPPORTED"
NOT_COMPARABLE = "NOT_COMPARABLE"

# (pt, en) per note code. Formatted with the entry's note_args.
NOTES = {
    "reference_absent": (
        "O relatório original ainda não foi extraído (report_reference.json ausente).",
        "The original report has not been transcribed yet (report_reference.json missing)."),
    "slot_not_transcribed": (
        "Esta seção não foi transcrita da foto do relatório original - não há com o que "
        "comparar (pode estar ilegível para o modelo de visão).",
        "This section was not transcribed from the photo of the original report - there is "
        "nothing to compare against (it may be illegible to the vision model)."),
    "both_blank": (
        "Nenhum dos dois relatórios preenche este campo.",
        "Neither report fills this field in."),
    "only_in_original": (
        "O relatório original preenche este campo; as evidências de campo não o sustentam.",
        "The original report fills this field in; the field evidence does not support it."),
    "only_in_original_expected": (
        "O relatório original preenche este campo; as evidências de campo não o sustentam. "
        "Lacuna estruturalmente esperada para este caso.",
        "The original report fills this field in; the field evidence does not support it. "
        "A structurally expected gap for this case."),
    "original_blank": (
        "O relatório original deixou esta seção em branco.",
        "The original report left this section blank."),
    "all_values_found": (
        "Todos os valores sustentados por evidência aparecem no relatório original.",
        "Every evidence-backed value also appears in the original report."),
    "partial_values_found": (
        "Correspondência parcial: {matched} de {total} valor(es) localizados no texto "
        "original; não localizados: {unmatched}.",
        "Partial match: {matched} of {total} value(s) found in the original text; "
        "not found: {unmatched}."),
    "no_values_found": (
        "Nenhum valor gerado foi localizado no texto original (pode ser diferença de "
        "redação, não necessariamente um erro).",
        "No generated value was found in the original text (this may be a difference in "
        "wording rather than an error)."),
    "no_checkable_claim": (
        "Ambos têm conteúdo, mas o texto gerado não faz afirmações pontuais a checar.",
        "Both sides have content, but the generated text makes no point claim to check."),
    "date_conflict": (
        "Data divergente: a evidência indica {expected}, o original traz {found}.",
        "Dates disagree: the evidence gives {expected}, the original gives {found}."),
    "area_supported": (
        "Área compatível com uma evidência observada.",
        "Area figure consistent with an observed piece of evidence."),
    "area_unsupported": (
        "Área citada no original sem correspondência em nenhuma evidência de campo.",
        "Area figure in the original with no match in any field evidence."),
    "document_supported": (
        "Número de documento confirmado pelas evidências.",
        "Document number confirmed by the evidence."),
    "document_unsupported": (
        "Número de documento do original sem confirmação nas evidências de campo.",
        "Document number in the original with no confirmation in the field evidence."),
    "document_unsupported_later": (
        "Número de documento do original sem confirmação nas evidências de campo - "
        "consistente com documentos emitidos após a fiscalização ({keys}).",
        "Document number in the original with no confirmation in the field evidence - "
        "consistent with documents issued after the inspection ({keys})."),
    "cpf_unsupported": (
        "CPF presente no original; nenhuma evidência de campo registra documentos de "
        "identidade.",
        "A CPF (tax ID) appears in the original; no field evidence records identity "
        "documents."),
    "person_supported": (
        "Pessoa também identificada nas evidências de campo.",
        "This person is also identified in the field evidence."),
    "person_unsupported": (
        "Pessoa nomeada no original sem correspondência em nenhuma evidência de campo.",
        "Person named in the original with no match in any field evidence."),
    "unsupported_claim": (
        "Alegação gerada que sua própria fonte citada não sustenta: {issue}",
        "A generated claim its own cited source does not support: {issue}"),
}

_HA_RE = re.compile(r"(\d{1,4}(?:[.,]\d{1,4})?)\s*ha\b", re.I)
# An area reported to one decimal ("23,4 ha") and the same measurement written
# to four in the original ("23.4187 ha") are the same area, not a discrepancy.
# One tolerance for both the per-section and the whole-text area checks, so the
# two halves of the comparison can never disagree about the same number.
_HA_TOLERANCE = 0.1
_DOC_NUMBER_RE = re.compile(r"\b\d{4,6}/\d{4}\b")
_CPF_RE = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
_NAMED_PERSON_RE = re.compile(
    r"\b(?:Sr\.?|Sra\.?|Senhor|Senhora)\s+"
    r"([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+){1,4})"
)

# A date is the one value type where "different from the evidence" always means
# disagreement: an occurrence has exactly one opening date. Areas and document
# numbers are deliberately NOT treated this way - a report legitimately cites
# several areas (total, estimate, suppressed) and several document numbers, so a
# number of the same shape elsewhere in the text is not evidence of a conflict.
_DATE_SHAPE_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")


def note_text(code: str, args: dict | None = None, lang: str = "pt") -> str:
    pair = NOTES.get(code)
    if pair is None:
        return code
    return pair[0 if lang == "pt" else 1].format(**(args or {}))


def _note(entry: dict, verdict: str, code: str, **args) -> dict:
    entry["verdict"] = verdict
    entry["note_code"] = code
    entry["note_args"] = args
    entry["note"] = note_text(code, args)  # pt-BR, so the JSON reads on its own
    return entry


def _format_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d/%m/%Y")
    except ValueError:
        return iso


def _ha_value(token: str) -> float | None:
    try:
        return float(token.replace(".", "").replace(",", ".")) if "," in token else float(token)
    except ValueError:
        return None


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _ha_figures(text: str) -> list[float]:
    """Every area figure in a piece of text, as numbers."""
    found = (_ha_value(m.group(1)) for m in _HA_RE.finditer(text or ""))
    return [v for v in found if v is not None]


def _areas_cited(clause: dict, facts_by_id: dict) -> list[float]:
    values = []
    for fid in clause.get("fact_ids", []):
        fact = facts_by_id.get(fid)
        if fact and fact["type"] == "area":
            try:
                values.append(float(str(fact["value"]).replace(",", ".")))
            except ValueError:
                pass
    return values


def _clause_matches_text(clause: dict, text_norm: str, original_areas: list[float],
                         facts_by_id: dict) -> bool:
    """Does this clause's value appear in the original text?

    The clause text already contains its fact's formatted value (guaranteed by
    generate_report.validate_report), so matching on tokens of the sentence is
    the same as matching on the value, without re-deriving it. Areas are the one
    exception: they are compared as numbers, because the two documents round the
    same measurement to different precision.
    """
    if clause["status"] != "supported":
        return False

    for area in _areas_cited(clause, facts_by_id):
        if any(abs(area - other) < _HA_TOLERANCE for other in original_areas):
            return True

    for token in re.findall(r"[\wÀ-ú][\wÀ-ú.,/\-]{2,}", clause["text"]):
        token = token.strip(".,;:")
        if not token or not re.search(r"[A-ZÀ-Ú0-9]", token):
            continue
        # on a word boundary, not anywhere in the text: "Mar" sits inside
        # "Deuzimar", and a substring hit there reported a whole section as
        # agreeing with the original on the strength of three letters
        if re.search(rf"\b{re.escape(normalize(token))}\b", text_norm):
            return True
    return False


def _date_conflict(clauses: list[dict], facts_by_id: dict, original_text: str):
    """A date in the original that differs from an observed opened_at fact."""
    for clause in clauses:
        for fid in clause.get("fact_ids", []):
            fact = facts_by_id.get(fid)
            if not fact or fact["type"] != "timestamp" or fact["status"] != "observed":
                continue
            expected = _format_date(fact["value"])
            found = sorted(set(_DATE_SHAPE_RE.findall(original_text)) - {expected})
            if found:
                return expected, ", ".join(found)
    return None


def _compare_slot(section: dict, original_text: str | None, facts_by_id: dict) -> dict:
    """One section of the form, generated side against original side.

    original_text is None in two distinguishable cases, and they must not be
    confused: the whole reference file is absent (nothing was ever read), or the
    reference exists but this slot is not in it (the vision model could not read
    that part of the page). Neither is evidence that the original left the field
    blank, so neither can be scored - only an empty string means "blank on the
    paper", and that can be a genuine MATCH.
    """
    entry = {
        "id": section["id"], "number": section.get("number"), "label": section["label"],
        "label_en": section["label_en"],
        "generated_status": section["status"],
        "generated_text": section["text"], "generated_text_en": section["text_en"],
        "original_text": original_text,
    }

    if original_text is None:
        return _note(entry, NOT_COMPARABLE, "slot_not_transcribed")

    generated_has = section["status"] in ("supported", "partial", "conflict")
    original_has = bool(original_text.strip())

    if not generated_has and not original_has:
        return _note(entry, MATCH, "both_blank")
    if not generated_has:
        return _note(entry, MISSING_FROM_GENERATED,
                     "only_in_original_expected" if section.get("expected_missing")
                     else "only_in_original")
    if not original_has:
        return _note(entry, NOT_COMPARABLE, "original_blank")

    clauses = [c for c in section.get("clauses", []) if c["status"] == "supported"]
    conflict = _date_conflict(clauses, facts_by_id, original_text)
    if conflict:
        return _note(entry, CONFLICT, "date_conflict", expected=conflict[0], found=conflict[1])
    if not clauses:
        return _note(entry, NOT_COMPARABLE, "no_checkable_claim")

    original_norm = normalize(original_text)
    original_areas = _ha_figures(original_text)
    matched = [c for c in clauses
               if _clause_matches_text(c, original_norm, original_areas, facts_by_id)]
    unmatched = [c for c in clauses if c not in matched]
    if not unmatched:
        return _note(entry, MATCH, "all_values_found")
    if matched:
        return _note(entry, PARTIAL_MATCH, "partial_values_found",
                     matched=len(matched), total=len(clauses),
                     unmatched=", ".join(c["key"] for c in unmatched))
    return _note(entry, EXTRA_IN_GENERATED, "no_values_found")


def _finding(kind: str, value: str, verdict: str, code: str, **args) -> dict:
    return _note({"kind": kind, "value_in_original": value}, verdict, code, **args)


def _generic_findings(reference_values: dict, facts: list[dict]) -> list[dict]:
    """Case-agnostic checks over the whole original text: values the evidence
    schema has no way to support at all, or that disagree with one it does.

    Nothing here is specific to Altamira - the same regexes apply to any
    municipality's printed report.
    """
    full_text = " ".join(v for v in reference_values.values() if v)
    if not full_text:
        return []

    findings = []
    # "inferred" counts as supported here: it is a value the report does state,
    # with its arithmetic shown. Leaving it out had the comparison call the
    # original's 20,9297 ha unsupported while the generated report was stating
    # 20,929 ha two lines above.
    observed = [f for f in facts if f["status"] in ("observed", "inferred")]

    supported_ha = set()
    for fact in (f for f in observed if f["type"] == "area"):
        try:
            supported_ha.add(float(str(fact["value"]).replace(",", ".")))
        except ValueError:
            pass
    for match in _HA_RE.finditer(full_text):
        value = _ha_value(match.group(1))
        if value is None:
            continue
        ok = any(abs(value - s) < _HA_TOLERANCE for s in supported_ha)
        findings.append(_finding("area_ha", f"{match.group(1)} ha",
                                 MATCH if ok else MISSING_FROM_GENERATED,
                                 "area_supported" if ok else "area_unsupported"))

    doc_values = {f["value"] for f in observed if f["type"] == "document"}
    issued_later = sorted({f["key"] for f in facts
                           if f["type"] == "document" and f["status"] == "missing"})
    for number in sorted(set(_DOC_NUMBER_RE.findall(full_text))):
        if number in doc_values:
            findings.append(_finding("document_number", number, MATCH, "document_supported"))
        elif issued_later:
            findings.append(_finding("document_number", number, MISSING_FROM_GENERATED,
                                     "document_unsupported_later", keys=", ".join(issued_later)))
        else:
            findings.append(_finding("document_number", number, MISSING_FROM_GENERATED,
                                     "document_unsupported"))

    for cpf in sorted(set(_CPF_RE.findall(full_text))):
        findings.append(_finding("cpf", cpf, MISSING_FROM_GENERATED, "cpf_unsupported"))

    people = {normalize(f["value"]) for f in observed if f["type"] == "person"}
    for name in sorted({m.group(1) for m in _NAMED_PERSON_RE.finditer(full_text)}):
        name_norm = normalize(name)
        ok = any(name_norm in person or person.split()[0] in name_norm.split()
                 for person in people)
        findings.append(_finding("named_person", name,
                                 MATCH if ok else MISSING_FROM_GENERATED,
                                 "person_supported" if ok else "person_unsupported"))

    return findings


def compare(report: dict, reference: dict | None, template: dict | None = None) -> dict:
    facts = report.get("_evidence_facts", [])
    facts_by_id = {f["id"]: f for f in facts}

    original_available = reference is not None
    reference_values = (reference or {}).get("values", {})

    slots = []
    for section in report["sections"]:
        if section["kind"] == "static":
            continue  # boilerplate printed on both forms: nothing to score
        if not original_available:
            slots.append(_note({
                "id": section["id"], "number": section.get("number"),
                "label": section["label"], "label_en": section["label_en"],
                "generated_status": section["status"],
                "generated_text": section["text"], "generated_text_en": section["text_en"],
                "original_text": None,
            }, NOT_COMPARABLE, "reference_absent"))
        else:
            slots.append(_compare_slot(section, reference_values.get(section["id"]), facts_by_id))

    generic = _generic_findings(reference_values, facts) if original_available else []
    unsupported = [
        _note({"kind": "unsupported_claim"}, UNSUPPORTED, "unsupported_claim", issue=issue)
        for issue in report.get("validation", {}).get("issues", [])
    ]

    counts: dict[str, int] = {}
    for entry in slots + generic + unsupported:
        counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1

    comparable = [s for s in slots if s["verdict"] != NOT_COMPARABLE]
    matches = sum(1 for s in comparable if s["verdict"] == MATCH)

    return {
        "case": report["case"],
        "original_available": original_available,
        "original_source": (reference or {}).get("source"),
        "slots": slots,
        "generic_findings": generic,
        "unsupported_claims": unsupported,
        "summary": counts,
        "completeness": {
            "total_sections": len(slots),
            "comparable_sections": len(comparable),
            "matching_sections": matches,
            "partial_sections": sum(1 for s in comparable if s["verdict"] == PARTIAL_MATCH),
            "ratio": round(matches / len(comparable), 3) if comparable else None,
        },
    }


def main(folder: Path) -> None:
    folder = Path(folder)
    report = load_json(folder / "output" / "report.json")
    if report is None:
        log.error("no output/report.json in %s - run generate_report.py first", folder)
        raise SystemExit(1)
    evidence = load_json(folder / "output" / "evidence.json")
    report["_evidence_facts"] = evidence["facts"] if evidence else []

    reference = load_json(folder / "report_reference.json")
    if reference is None:
        log.warning("no report_reference.json in %s - every section will be reported as "
                    "not comparable rather than scored against a fabricated original; run "
                    "extract_from_final_report.py to produce it", folder)

    result = compare(report, reference)

    out_path = folder / "output" / "comparison.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("%s: %s", folder.name,
             ", ".join(f"{n} {v}" for v, n in sorted(result["summary"].items())))
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("folder", nargs="?", default=".")
    args = parser.parse_args()

    main(Path(args.folder))
