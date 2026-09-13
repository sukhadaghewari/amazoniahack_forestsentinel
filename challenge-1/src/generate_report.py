"""
Assemble the Relatório de Fiscalização from data/<case>/output/evidence.json -
never from raw evidence, and never from report_reference.json (that file is the
answer document; see extract_from_final_report.py).

One slot of the printed form becomes one section. A section is built from named
clauses, one per fact key the slot draws on, and a clause has exactly two
possible shapes:

  - it states a value, quoting it from a fact whose status is "observed" or
    "inferred", or
  - it states plainly that the evidence does not support the field.

There is no third shape. A value is never guessed, and two disagreeing readings
are never collapsed into one - see _clause_conflict.

Both languages come out of the same facts: every clause carries `text` (pt-BR,
the legal wording) and `text_en`, rendered from the same fact by the same
template pair. The English view is therefore a re-render, not a translation -
no model is involved and the values cannot drift between the two.

Drafting calls no model at all. Every sentence is rendered from a fact by a
template, so there is no decoding step in which the report could say something
the evidence does not. An earlier --polish flag asked a local text model to
rewrite section 8 as prose; it once filed the model's entire chain of thought
as that section, and it is gone.

Usage:
    python generate_report.py data/altamira
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from facts import normalize, relative_file
from logconf import setup

log = setup("generate_report")

LANGS = ("pt", "en")

MISSING_TEXT = {
    "pt": "Não informado nos elementos de evidência disponíveis.",
    "en": "Not stated in the available field evidence.",
}

# The printed form: which evidence.json fact fills each numbered section.
# "static" slots are boilerplate printed on the form itself - not claims about
# this case, so they carry no fact and need no source. Every other slot names
# the keys build_evidence.py produces; a slot whose keys are all absent or
# "missing" stays explicitly empty rather than being invented.
#
# Written out by hand from the paper form rather than read off the photograph,
# so that a VLM misreading a heading cannot corrupt the structure every later
# step depends on.
REPORT_SLOTS = [
    {
        "id": "solicitante", "number": 1,
        "label": "Solicitante", "label_en": "Requesting body",
        "kind": "static",
        "value": "Secretaria Municipal da Gestão do Meio Ambiente - SEMMA.",
        "value_en": "Municipal Secretariat for Environmental Management - SEMMA.",
        "fills_from": [],
    },
    {
        "id": "autor", "number": 2,
        "label": "Autor", "label_en": "Author",
        "kind": "fact", "fills_from": ["inspector_full_name"],
    },
    {
        "id": "local", "number": 3,
        "label": "Local e Abrangência da Ação", "label_en": "Location and extent",
        "kind": "composite", "fills_from": ["title", "perimeter_points"],
        # The paper report cites a single point. No recorded fix matches it, so
        # the opening fix of the occurrence is used: a real reading, observed
        # rather than computed, at a defensible place - the moment work started.
        "coordinate_rule": "first_perimeter_point",
    },
    {
        "id": "objetivo", "number": 4,
        "label": "Objetivo", "label_en": "Purpose",
        "kind": "composite", "fills_from": ["category"],
    },
    {
        "id": "data", "number": 5,
        "label": "Data", "label_en": "Date",
        "kind": "fact", "fills_from": ["opened_at"],
    },
    {
        "id": "horario", "number": 6,
        "label": "Horário", "label_en": "Time of arrival",
        "kind": "fact", "fills_from": ["arrival_time"],
        # Nothing in the field evidence records an arrival time. Audio 01 is
        # stamped 07:52 at a point ~8 km short of the site and says "saiu da
        # sede seis e quarenta"; the next record is the occurrence opening at
        # 08:26, and audio 02 says "8 e 27 [...] Chegamos". The team spent the
        # first ~24 minutes on the abordagem, recording nothing, so the paper
        # report's 08h02min is the agent's own recollection and cannot be
        # derived. Checked 2026-09-13: the 07:52-08:26 window does bracket it,
        # but the field stays empty on purpose - Horario wants one clock time,
        # not an interval - and the spoken 06h40 is deliberately not extracted.
        "expected_missing": True,
    },
    {
        "id": "equipe", "number": 7,
        "label": "Equipe", "label_en": "Team",
        "kind": "list", "fills_from": ["team_full_names", "team_member"],
    },
    {
        "id": "metodologia", "number": 8,
        "label": "Metodologia e Descrição das atividades",
        "label_en": "Methodology and description of activities",
        "kind": "narrative",
        "fills_from": [
            "category", "opened_at", "title", "car_registry",
            "operator_name", "owner_name", "equipment_trator", "equipment_motosserra",
            "auto_constatacao_number", "infraction_notice_number", "embargo_notice_number",
            "total_suppressed_area_ha", "app_area_ha", "reserve_legal_suppressed_ha",
            "watercourse_name", "area_estimate_ha",
        ],
    },
    {
        "id": "enquadramento", "number": None,
        "label": "Enquadramento legal", "label_en": "Legal basis",
        "kind": "fact", "fills_from": ["legal_citation"],
    },
]

# One (pt, en) label per fact key. Keys absent here fall back to the key itself.
_LABELS = {
    "category": ("Classificação da ocorrência", "Occurrence classification"),
    "opened_at": ("Data e horário de abertura", "Date and time opened"),
    "title": ("Local", "Location"),
    "perimeter_points": ("Ponto de referência", "Reference point"),
    "car_registry": ("Registro CAR do imóvel", "CAR registration of the property"),
    "operator_name": ("Operador(a) do equipamento identificado no local",
                      "Operator of the equipment identified on site"),
    "owner_name": ("Responsável/proprietário(a)", "Person responsible / owner"),
    "equipment_trator": ("Equipamento encontrado (trator)", "Equipment found (tractor)"),
    "equipment_motosserra": ("Equipamento encontrado (motosserra)", "Equipment found (chainsaw)"),
    "auto_constatacao_number": ("Auto de Constatação", "Finding Notice"),
    "infraction_notice_number": ("Auto de Infração", "Infraction Notice"),
    "embargo_notice_number": ("Auto de Embargo", "Embargo Notice"),
    "total_suppressed_area_ha": ("Área total suprimida", "Total cleared area"),
    "app_area_ha": ("Área suprimida em Área de Preservação Permanente (APP)",
                    "Area cleared in a Permanent Preservation Area (APP)"),
    "reserve_legal_suppressed_ha": ("Área suprimida em Reserva Legal",
                                    "Area cleared in the Legal Reserve"),
    "watercourse_name": ("Curso d'água afetado", "Affected watercourse"),
    "area_estimate_ha": ("Área estimada da ocorrência (registro do aplicativo)",
                         "Area of the occurrence estimated by the app"),
    "inspector_full_name": ("Fiscal responsável (nome completo)",
                            "Inspector in charge (full name)"),
    "arrival_time": ("Horário de chegada da equipe", "Team arrival time"),
    "team_full_names": ("Integrantes da equipe (nomes completos)", "Team members (full names)"),
    "legal_citation": ("Enquadramento legal", "Legal basis"),
    # not used by any slot of the form, but extracted and shown in the
    # evidence view, so they need a label like everything else
    "property_area_ha": ("Área total do imóvel", "Total area of the property"),
    "reserve_legal_area_ha": ("Área de Reserva Legal do imóvel",
                              "Legal Reserve area of the property"),
    "report_number": ("Número do relatório", "Report number"),
    "inspector_name": ("Nome do fiscal como registrado", "Inspector's name as recorded"),
    "team_member": ("Integrante da equipe", "Team member"),
    "burned_area_estimate_ha": ("Estimativa da área queimada", "Estimated burned area"),
    "signature_refused": ("Recusa de assinatura do autuado",
                          "The cited party refused to sign"),
}

# One (pt, en) sentence per fact key. Keys absent here fall back to
# "{label}: {value}." in both languages.
_TEMPLATES = {
    "category": ("A ocorrência foi classificada como {value}.",
                 "The occurrence is classified as {value}."),
    "opened_at": ("A ocorrência foi aberta em {date} às {time}.",
                  "The occurrence was opened on {date} at {time}."),
    "title": ("Local da ação: {value}.", "Location of the action: {value}."),
    "car_registry": ("O imóvel está registrado no CAR sob o número {value}.",
                     "The property is registered in the CAR under number {value}."),
    "operator_name": ("{value} foi identificado(a) no local como operador(a) "
                      "do equipamento encontrado em atividade.",
                      "{value} was identified on site as the operator of the "
                      "equipment found in operation."),
    "owner_name": ("Responsável/proprietário(a) identificado(a) nas evidências: {value}.",
                   "Person responsible / owner identified in the evidence: {value}."),
    "equipment_trator": ("Foi encontrado no local um trator {value}.",
                         "A {value} tractor was found on site."),
    "equipment_motosserra": ("Foi encontrada no local uma motosserra {value}.",
                             "A {value} chainsaw was found on site."),
    "auto_constatacao_number": ("Foi lavrado o Auto de Constatação nº {value}.",
                                "Finding Notice (Auto de Constatação) nº {value} was issued."),
    "infraction_notice_number": ("Auto de Infração nº {value}.",
                                 "Infraction Notice (Auto de Infração) nº {value}."),
    "embargo_notice_number": ("Auto de Embargo nº {value}.",
                              "Embargo Notice (Auto de Embargo) nº {value}."),
    "total_suppressed_area_ha": ("A área total suprimida é de {value} ha.",
                                 "The total cleared area is {value} ha."),
    "app_area_ha": ("A área suprimida em Área de Preservação Permanente (APP) é de {value} ha.",
                    "The area cleared in a Permanent Preservation Area (APP) is {value} ha."),
    "reserve_legal_suppressed_ha": ("A área suprimida em Reserva Legal é de {value} ha "
                                    "(valor calculado, não medido).",
                                    "The area cleared in the Legal Reserve is {value} ha "
                                    "(calculated, not measured)."),
    "watercourse_name": ("O curso d'água afetado identificado é: {value}.",
                         "The affected watercourse identified is: {value}."),
    "area_estimate_ha": ("A área estimada da ocorrência, conforme registro do aplicativo, "
                         "é de {value} ha.",
                         "The area of the occurrence, as recorded by the app, is {value} ha."),
}

# A list slot whose authoritative key (full names) has no evidence, but whose
# fallback key (first names as recorded) does. Printing the first names as if
# they answered the section would overstate them; printing nothing would throw
# away evidence the report should carry. So it states both, and counts as
# partial.
_PARTIAL_LIST_TEXT = (
    "Nomes completos não informados nos elementos de evidência disponíveis. "
    "Primeiros nomes registrados nas evidências: {values}.",
    "Full names are not stated in the available field evidence. First names "
    "recorded in the evidence: {values}.",
)

_EXPECTED_MISSING_NOTE = (
    "Campo estruturalmente ausente das evidências de campo deste caso - "
    "não é uma falha de extração.",
    "Structurally absent from this case's field evidence - not an extraction failure.",
)


def _pick(pair, lang: str) -> str:
    return pair[LANGS.index(lang)]


def label(key: str, lang: str) -> str:
    return _pick(_LABELS[key], lang) if key in _LABELS else key


def _format_ha(value, lang: str) -> str:
    """23.4 -> '23,4' in pt-BR, '23.4' in English. No invented precision."""
    try:
        number = float(str(value).replace(",", "."))
    except ValueError:
        return str(value)
    formatted = f"{number:g}"
    return formatted.replace(".", ",") if lang == "pt" else formatted


def _format_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d/%m/%Y")
    except ValueError:
        return iso


def _format_time(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%H:%M")
    except ValueError:
        return iso


def _capitalize_brand(text: str) -> str:
    """'komatsu D51' -> 'Komatsu D51'. Presentation only, and only where the
    source wrote a brand in lower case because it was typed one-handed in the
    field; a word that already carries capitals ("MS382") is left alone."""
    return " ".join(word if any(c.isupper() for c in word) else word.capitalize()
                    for word in text.split())


def format_value(key: str, fact: dict, lang: str = "pt") -> str:
    if fact["type"] == "area":
        return _format_ha(fact["value"], lang)
    if fact["type"] == "equipment":
        return _capitalize_brand(str(fact["value"]))
    return str(fact["value"])


def load_evidence(folder: Path) -> dict:
    path = folder / "output" / "evidence.json"
    if not path.exists():
        log.error("no %s - run build_evidence.py first", path)
        raise SystemExit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def _index_facts(evidence: dict) -> dict[str, list[dict]]:
    """key -> every fact under that key (usually one; team_member repeats)."""
    by_key: dict[str, list[dict]] = {}
    for f in evidence["facts"]:
        by_key.setdefault(f["key"], []).append(f)
    return by_key


# --------------------------------------------------------------------------
# clauses: one fact key -> one traceable sentence, in both languages
# --------------------------------------------------------------------------

def _clause_missing(key: str) -> dict:
    return {
        "key": key, "status": "missing", "fact_ids": [],
        "text": f"{label(key, 'pt')}: {MISSING_TEXT['pt']}",
        "text_en": f"{label(key, 'en')}: {MISSING_TEXT['en']}",
    }


def _clause_conflict(key: str, fact: dict) -> dict:
    """Both readings, side by side - never one picked silently for a report."""
    variants = fact.get("variants", [{"value": fact["value"],
                                      "confidence": fact["confidence"]}])
    texts = {}
    for lang in LANGS:
        readings = "; ".join(
            f"{format_value(key, {**fact, 'value': v['value']}, lang)!r} "
            f"({'confiança' if lang == 'pt' else 'confidence'} {v['confidence']})"
            for v in variants
        )
        body = ("as fontes de evidência divergem e a divergência não foi resolvida "
                f"automaticamente. Leituras registradas: {readings}."
                if lang == "pt" else
                "the evidence sources disagree and the disagreement was not resolved "
                f"automatically. Readings recorded: {readings}.")
        texts[lang] = f"{label(key, lang)}: {body}"
    return {"key": key, "status": "conflict", "fact_ids": [fact["id"]],
            "text": texts["pt"], "text_en": texts["en"]}


def _clause(key: str, by_key: dict[str, list[dict]]) -> dict:
    facts = by_key.get(key)
    if not facts or facts[0]["status"] == "missing":
        return _clause_missing(key)

    fact = facts[0]
    if fact["status"] == "conflicting":
        return _clause_conflict(key, fact)

    texts = {}
    for lang in LANGS:
        template = _pick(_TEMPLATES[key], lang) if key in _TEMPLATES else "{label}: {value}."
        if key == "opened_at":
            texts[lang] = template.format(date=_format_date(fact["value"]),
                                          time=_format_time(fact["value"]))
        elif "{label}" in template:
            texts[lang] = template.format(label=label(key, lang),
                                          value=format_value(key, fact, lang))
        else:
            texts[lang] = template.format(value=format_value(key, fact, lang))

    return {"key": key, "status": "supported", "fact_ids": [fact["id"]],
            "confidence": fact["confidence"], "text": texts["pt"], "text_en": texts["en"]}


def _clauses_for_list_key(key: str, by_key: dict[str, list[dict]]) -> list[dict]:
    """team_member-shaped keys: zero or more repeated facts, each its own clause."""
    facts = [f for f in by_key.get(key, []) if f["status"] != "missing"]
    if not facts:
        return [_clause_missing(key)]
    return [{"key": key, "status": "supported", "fact_ids": [f["id"]],
             "confidence": f["confidence"],
             "text": str(f["value"]), "text_en": str(f["value"])}
            for f in facts]


# --------------------------------------------------------------------------
# sections: one form slot -> its clauses, its status, its rendered text
# --------------------------------------------------------------------------

def _status(clauses: list[dict]) -> str:
    statuses = {c["status"] for c in clauses}
    if statuses == {"supported"}:
        return "supported"
    if statuses == {"missing"}:
        return "missing"
    if "conflict" in statuses:
        return "conflict"
    return "partial"


def _section(slot: dict, clauses: list[dict], status: str | None = None,
             texts: tuple[str, str] | None = None) -> dict:
    return {
        "id": slot["id"], "number": slot["number"],
        "label": slot["label"], "label_en": slot["label_en"],
        "kind": slot["kind"],
        "status": status or _status(clauses),
        "text": texts[0] if texts else " ".join(c["text"] for c in clauses),
        "text_en": texts[1] if texts else " ".join(c["text_en"] for c in clauses),
        "clauses": clauses,
        "expected_missing": bool(slot.get("expected_missing")),
    }


def _build_static(slot: dict, by_key: dict) -> dict:
    return _section(slot, [], status="static", texts=(slot["value"], slot["value_en"]))


def _build_fact(slot: dict, by_key: dict) -> dict:
    clause = _clause(slot["fills_from"][0], by_key)
    if slot.get("expected_missing") and clause["status"] == "missing":
        clause["note"], clause["note_en"] = _EXPECTED_MISSING_NOTE
    return _section(slot, [clause])


def _build_composite(slot: dict, by_key: dict) -> dict:
    if slot["id"] == "local":
        points = by_key.get("perimeter_points")
        if points and points[0]["status"] != "missing":
            lat, lon = points[0]["value"][0]
            point = {
                "key": "perimeter_points", "status": "supported",
                "fact_ids": [points[0]["id"]], "confidence": points[0]["confidence"],
                "text": (f"Ponto de referência (primeiro ponto marcado no início da "
                         f"ocorrência): {lat}, {lon}."),
                "text_en": (f"Reference point (first point marked when the occurrence "
                            f"was opened): {lat}, {lon}."),
                # coordinate_set has no single scalar for validate_report to re-check
                "skip_value_check": True,
            }
        else:
            point = _clause_missing("perimeter_points")
        return _section(slot, [_clause("title", by_key), point])

    # objetivo: the classification, phrased as the purpose of the inspection
    clause = _clause("category", by_key)
    if clause["status"] == "supported":
        value = by_key["category"][0]["value"]
        clause = {**clause,
                  "text": f"Fiscalização ambiental referente à ocorrência classificada "
                          f"como: {value}.",
                  "text_en": f"Environmental inspection of the occurrence classified "
                             f"as: {value}."}
    return _section(slot, [clause])


def _build_list(slot: dict, by_key: dict) -> dict:
    """A slot filled from several facts of the same kind - the team's members.

    fills_from is ordered: the first key answers the section outright (the full
    names the form asks for), the rest only partly (the first names the
    evidence actually recorded).
    """
    authoritative, *fallbacks = slot["fills_from"]
    clauses = [_clause(authoritative, by_key)]
    for key in fallbacks:
        clauses += _clauses_for_list_key(key, by_key)

    if clauses[0]["status"] == "supported":
        supported = [c for c in clauses if c["status"] == "supported"]
        return _section(slot, clauses, status="supported", texts=(
            "; ".join(c["text"] for c in supported) + ".",
            "; ".join(c["text_en"] for c in supported) + "."))

    partial = [c for c in clauses[1:] if c["status"] == "supported"]
    if partial:
        return _section(slot, clauses, status="partial", texts=tuple(
            _pick(_PARTIAL_LIST_TEXT, lang).format(
                values="; ".join(c["text" if lang == "pt" else "text_en"] for c in partial))
            for lang in LANGS))

    return _section(slot, clauses, status="missing",
                    texts=(clauses[0]["text"], clauses[0]["text_en"]))


def _build_narrative(slot: dict, by_key: dict) -> dict:
    return _section(slot, [_clause(key, by_key) for key in slot["fills_from"]])


_BUILDERS = {
    "static": _build_static,
    "fact": _build_fact,
    "composite": _build_composite,
    "list": _build_list,
    "narrative": _build_narrative,
}


# --------------------------------------------------------------------------
# Self-check, run on every generate()
# --------------------------------------------------------------------------


def validate_report(report: dict) -> dict:
    """Defense in depth: every 'supported' clause must cite a real fact whose
    value actually appears in the sentence printed for it.

    This cannot fire by construction - the sentence is built from the fact it
    cites - which is the point: it is the backstop that catches a future bug
    in a template or a builder before a report ships.
    """
    facts_by_id = {f["id"]: f for f in report["_evidence_facts"]}
    issues = []

    for section in report["sections"]:
        for clause in section.get("clauses", []):
            if clause["status"] != "supported":
                continue
            if not clause["fact_ids"]:
                issues.append(f"{section['id']}/{clause['key']}: supported with no fact_ids")
                continue
            if clause.get("skip_value_check"):
                continue
            for fid in clause["fact_ids"]:
                fact = facts_by_id.get(fid)
                if fact is None:
                    issues.append(f"{section['id']}/{clause['key']}: cites unknown fact {fid}")
                    continue
                if clause["key"] == "opened_at":
                    expected = [_format_date(fact["value"]), _format_time(fact["value"])]
                else:
                    expected = [format_value(clause["key"], fact)]
                text = normalize(clause["text"])
                if any(e and normalize(e) not in text for e in expected):
                    issues.append(f"{section['id']}/{clause['key']}: fact {fid} value "
                                  f"{fact['value']!r} not found in its own clause text")

    dates = {f["value"][:10] for f in report["_evidence_facts"]
             if f["type"] == "timestamp" and f["status"] == "observed" and f["value"]}
    if len(dates) > 1:
        issues.append(f"inconsistent dates across observed facts: {sorted(dates)}")

    return {"ok": not issues, "issues": issues}


def _section_confidence(section: dict, facts_by_id: dict) -> float | None:
    """The weakest confidence any of this section's claims rests on.

    The weakest, not the average: a paragraph is only as sound as its shakiest
    sentence, and a reviewer deciding whether to trust one needs to know that a
    single figure in it came off a voice note.
    """
    scores = [facts_by_id[fid]["confidence"]
              for clause in section["clauses"] for fid in clause["fact_ids"]
              if fid in facts_by_id]
    return min(scores) if scores else None


def _section_sources(section: dict, facts_by_id: dict) -> list[str]:
    """Every evidence file this section's claims rest on, once each.

    One line per section is all an auditor needs to see at a glance; the full
    per-clause citation stays in the clauses themselves.
    """
    files = set()
    for clause in section["clauses"]:
        for fid in clause["fact_ids"]:
            for source in facts_by_id.get(fid, {}).get("sources", []):
                files.add(relative_file(source))
    return sorted(f for f in files if f)


def generate(folder: Path) -> dict:
    evidence = load_evidence(folder)
    by_key = _index_facts(evidence)
    facts_by_id = {f["id"]: f for f in evidence["facts"]}

    sections = [_BUILDERS[slot["kind"]](slot, by_key) for slot in REPORT_SLOTS]

    counts: dict[str, int] = {}
    for section in sections:
        section["sources"] = _section_sources(section, facts_by_id)
        section["confidence"] = _section_confidence(section, facts_by_id)
        counts[section["status"]] = counts.get(section["status"], 0) + 1

    report = {
        "case": evidence["case"],
        "document": "Relatório de Fiscalização",
        "document_en": "Enforcement Inspection Report",
        "issuer": "SEMMA - Secretaria Municipal da Gestão do Meio Ambiente, Altamira/PA",
        "language": "pt-BR",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sections": sections,
        "summary": counts,
        "_evidence_facts": evidence["facts"],
    }

    report["validation"] = validate_report(report)
    if not report["validation"]["ok"]:
        log.error("report failed internal validation: %s", report["validation"]["issues"])

    return report


def public(report: dict) -> dict:
    """The report without the evidence it was validated against - what gets
    written to disk and returned over HTTP."""
    return {k: v for k, v in report.items() if k != "_evidence_facts"}


def render_text(report: dict, lang: str = "pt") -> str:
    """The report as a person reads it: plain text, in the order of the form."""
    document = report["document"] if lang == "pt" else report["document_en"]
    lines = [document, report["issuer"], ""]
    for section in report["sections"]:
        label = section["label"] if lang == "pt" else section["label_en"]
        heading = f"{section['number']}. {label}" if section["number"] else label
        lines += [heading, section["text"] if lang == "pt" else section["text_en"], ""]
    return "\n".join(lines)


def main(folder: Path) -> None:
    folder = Path(folder)
    report = generate(folder)

    out_dir = folder / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(public(report), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.txt").write_text(render_text(report), encoding="utf-8")

    log.info("%s: %s", folder.name,
             ", ".join(f"{n} {s}" for s, n in sorted(report["summary"].items())))
    log.info("wrote %s/report.json and %s/report.txt", out_dir, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("folder", nargs="?", default=".")
    args = parser.parse_args()

    main(Path(args.folder))
