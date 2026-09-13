"""
FastAPI surface over the pipeline. Every endpoint reads or writes the same
data/<case>/ files the command-line scripts use - there is no database and
no hidden state, so `python build_evidence.py data/altamira` and
`POST /process-case` produce the exact same output/evidence.json.

What this API does NOT do: it never calls an extractor that loads Whisper,
PaddleOCR or the Qwen3-VL vision model (extract_transcriptions.py,
extract_text_from_photos.py, extract_from_field_notes.py,
extract_from_occurrence_summary.py, extract_from_final_report.py). Those are
run once, offline, by whoever operates the pipeline, and their output is
just read from data/<case>/.cache/ and data/<case>/report_reference.json.
This keeps every request here fast and side-effect-free on the expensive
part of the system, and means a request never silently kicks off a model
load a caller didn't ask for.

If a prerequisite file is missing, the endpoint returns 409 with exactly
which command to run - never a fabricated result and never a silent
fallback to something plausible-looking.

pip install fastapi "uvicorn[standard]"

Usage:
    uvicorn api:app --reload --port 8000
"""

import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

import compare_report
import generate_report
from build_evidence import build as build_evidence
from logconf import setup

log = setup("api")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TAGS = [
    {"name": "cases", "description": "What evidence exists for a case, and how far the "
                                     "pipeline has got with it."},
    {"name": "facts", "description": "The structured, source-attributed facts the report "
                                     "is written from - evidence.json as it stands."},
    {"name": "report", "description": "The Portuguese inspection report, assembled from "
                                      "validated facts only."},
    {"name": "comparison", "description": "The generated report against the one the "
                                          "municipality filed, field by field."},
    {"name": "files", "description": "The raw evidence itself, served so a claim can be "
                                     "shown next to its source."},
]

app = FastAPI(
    title="Relatório de Fiscalização - evidence-to-report pipeline",
    version="1.0.0",
    openapi_tags=TAGS,
    description=(
        "Field evidence → extracted facts → Portuguese inspection report → comparison "
        "against the report the municipality filed → claim-level traceability.\n\n"
        "**Every claim in the generated report is either backed by a cited fact or "
        "explicitly marked as not stated in the evidence.** Nothing is inferred from what "
        "a report of this kind usually says.\n\n"
        "Everything runs locally: faster-whisper for the voice notes, PP-OCRv5 for the "
        "stamps burned into the photos, qwen3-vl:4b via Ollama for the free text. Drafting "
        "the report calls no model at all - it is deterministic from the facts, which is "
        "why the same evidence always produces the same report.\n\n"
        "This service never runs an extractor itself. The model-loading steps are run "
        "once, offline, by whoever operates the pipeline, and their output is read from "
        "`data/<case>/.cache/`. If a prerequisite file is missing you get a 409 naming the "
        "exact command to run - never a fabricated result."
    ),
)

# Trimmed from real output for this case. Enough to see the shape of a response
# without pasting a 20 KB document into the docs page.
_FACT_EXAMPLE = {
    "id": "fact_012",
    "key": "total_suppressed_area_ha",
    "type": "area",
    "value": "23,418",
    "status": "observed",
    "confidence": 0.62,
    "sources": [{
        "modality": "audio", "file": "09.mp3", "start": 11.27,
        "text": "Área total suprimida 23,418 hectares, sendo 2,489 área de preservação permanente.",
    }],
}

_SECTION_EXAMPLE = {
    "id": "objetivo", "number": 4, "label": "Objetivo", "label_en": "Purpose",
    "kind": "composite", "status": "supported",
    "text": "Fiscalização ambiental referente à ocorrência classificada como: Desmatamento.",
    "text_en": "Environmental inspection of the occurrence classified as: Desmatamento.",
    "clauses": [{
        "key": "category", "status": "supported", "fact_ids": ["fact_004"],
        "confidence": 0.99,
        "text": "Fiscalização ambiental referente à ocorrência classificada como: Desmatamento.",
        "text_en": "Environmental inspection of the occurrence classified as: Desmatamento.",
    }],
    "expected_missing": False,
    "sources": ["occurrence-summary.txt"],
    "confidence": 0.99,
}


def _example(payload: dict, description: str) -> dict:
    """One 200 response with a worked example, for the docs page."""
    return {200: {"description": description,
                  "content": {"application/json": {"example": payload}}}}

_CASE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_EVIDENCE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mp3", ".m4a", ".wav", ".ogg", ".md", ".txt"}


def _case_dir(case_id: str) -> Path:
    if not _CASE_ID_RE.match(case_id):
        raise HTTPException(400, f"invalid case id: {case_id!r}")
    folder = DATA_DIR / case_id
    if not folder.is_dir():
        raise HTTPException(404, f"no such case: {case_id!r} (looked in {folder})")
    return folder


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _require_json(path: Path, how_to_produce: str) -> dict:
    data = _read_json(path)
    if data is None:
        raise HTTPException(409, f"missing {path.relative_to(DATA_DIR.parent)} - run: {how_to_produce}")
    return data


@app.get("/", tags=["cases"], summary="What this service exposes",
         responses=_example({"service": "amazoniahack challenge-1 report pipeline",
                             "endpoints": ["GET  /cases", "..."]},
                            "The endpoint list, for a quick check that the service is up."))
def root():
    return {
        "service": "amazoniahack challenge-1 report pipeline",
        "endpoints": [
            "GET  /cases",
            "POST /process-case",
            "POST /extract-facts",
            "POST /generate-report",
            "POST /compare-report",
            "GET  /case/{case_id}",
            "GET  /case/{case_id}/traceability",
            "GET  /case/{case_id}/evidence/{relpath}",
        ],
    }


@app.get("/cases", tags=["cases"], summary="List the municipalities with a case folder",
         responses=_example({"cases": ["altamira"]}, "Every case under data/."))
def list_cases():
    if not DATA_DIR.is_dir():
        return {"cases": []}
    return {"cases": sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name != ".DS_Store")}


@app.post("/process-case", tags=["facts"],
          summary="Consolidate the extractors' output into one evidence file",
          responses=_example(
              {"case": "altamira",
               "summary": {"observed": 40, "inferred": 1, "missing": 5},
               "fact_count": 46},
              "What the consolidation produced. Deterministic: same cache, same facts."))
def process_case(case_id: str):
    """Rebuild output/evidence.json from whatever is already in .cache/.

    Deterministic consolidation only - see build_evidence.py's own docstring.
    Does not run any extractor; if .cache/ is empty this will simply produce
    a mostly-empty evidence set rather than an error, because that is an
    honest description of a case with no field evidence processed yet.
    """
    folder = _case_dir(case_id)
    if not (folder / ".cache").is_dir():
        raise HTTPException(
            409,
            f"no .cache/ for {case_id!r} - run the extractors first "
            "(extract_from_occurrence_summary.py, extract_text_from_photos.py, "
            "extract_from_field_notes.py), then retry",
        )

    evidence = build_evidence(folder)
    out_path = folder / "output" / "evidence.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("process-case %s: %s", case_id, evidence["summary"])
    return {"case": case_id, "summary": evidence["summary"], "fact_count": len(evidence["facts"])}


@app.post("/extract-facts", tags=["facts"],
          summary="The facts behind the report, with their sources and confidence",
          responses=_example(
              {"case": "altamira",
               "facts": [_FACT_EXAMPLE],
               "summary": {"observed": 40, "inferred": 1, "missing": 5}},
              "Every fact, each with every passage that backs it. A field the evidence "
              "does not support is present with value null and status \"missing\" - never "
              "absent, never guessed."))
def extract_facts(case_id: str):
    """The structured, source-attributed facts behind the report - evidence.json as is."""
    folder = _case_dir(case_id)
    evidence = _require_json(folder / "output" / "evidence.json",
                             f"POST /process-case?case_id={case_id}")
    return evidence


@app.post("/generate-report", tags=["report"],
          summary="Draft the report from validated facts only",
          responses=_example(
              {"case": "altamira", "document": "Relatório de Fiscalização",
               "document_en": "Enforcement Inspection Report",
               "issuer": "SEMMA - Secretaria Municipal da Gestão do Meio Ambiente, Altamira/PA",
               "language": "pt-BR", "generated_at": "2026-03-12T14:03:11",
               "sections": [_SECTION_EXAMPLE],
               "summary": {"static": 1, "supported": 4, "partial": 2, "missing": 2},
               "validation": {"ok": True, "issues": []}},
              "Every section in both languages, rendered from the same facts - the English "
              "is a re-render, not a translation, so no value can differ between them. "
              "validation.ok false means a clause cites a fact whose value is not actually "
              "in the sentence printed for it."))
def generate_report_endpoint(case_id: str):
    """Build the Portuguese report from validated facts only.

    Calls no model: every sentence is rendered from a cited fact by a
    template, so the same evidence always produces the same report.
    """
    folder = _case_dir(case_id)
    if not (folder / "output" / "evidence.json").exists():
        raise HTTPException(409, f"missing output/evidence.json - run: "
                                 f"POST /process-case?case_id={case_id}")

    report = generate_report.generate(folder)

    (folder / "output").mkdir(parents=True, exist_ok=True)
    to_write = generate_report.public(report)
    (folder / "output" / "report.json").write_text(
        json.dumps(to_write, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "output" / "report.txt").write_text(
        generate_report.render_text(report), encoding="utf-8")

    if not report["validation"]["ok"]:
        log.error("generate-report %s failed internal validation: %s",
                  case_id, report["validation"]["issues"])
    return to_write


@app.post("/compare-report", tags=["comparison"],
          summary="The generated report against the filed one, field by field",
          responses=_example(
              {"case": "altamira", "original_available": True,
               "original_source": "data/altamira/final_report.jpg",
               "slots": [{"id": "data", "number": 5, "label": "Data", "label_en": "Date",
                          "generated_status": "supported",
                          "generated_text": "A ocorrência foi aberta em 12/03/2026 às 08:26.",
                          "generated_text_en": "The occurrence was opened on 12/03/2026 at 08:26.",
                          "original_text": "12/03/2026", "verdict": "MATCH",
                          "note_code": "all_values_found", "note_args": {},
                          "note": "Todos os valores sustentados por evidência aparecem no "
                                  "relatório original."}],
               "generic_findings": [], "unsupported_claims": [],
               "summary": {"MATCH": 7, "PARTIAL_MATCH": 2, "MISSING_FROM_GENERATED": 5,
                           "NOT_COMPARABLE": 4},
               "completeness": {"total_sections": 8, "comparable_sections": 4,
                                "matching_sections": 1, "partial_sections": 2, "ratio": 0.25}},
              "One of seven verdicts per section, plus case-agnostic checks over the whole "
              "original text (areas, document numbers, CPFs, named people). "
              "MISSING_FROM_GENERATED is frequently the correct outcome, not a failure: the "
              "evidence really does not support the field. Render note_code in either "
              "language with compare_report.note_text()."))
def compare_report_endpoint(case_id: str):
    """Generated report vs. the original filled report, field by field.

    Never fabricates the original side: if report_reference.json does not
    exist (extract_from_final_report.py has not been run for this case),
    every slot comes back NOT_COMPARABLE with original_available=false
    rather than silently comparing against nothing.
    """
    folder = _case_dir(case_id)
    report = _read_json(folder / "output" / "report.json")
    if report is None:
        raise HTTPException(409, f"missing output/report.json - run: "
                                 f"POST /generate-report?case_id={case_id}")
    evidence = _require_json(folder / "output" / "evidence.json",
                             f"POST /process-case?case_id={case_id}")
    report["_evidence_facts"] = evidence["facts"]

    reference = _read_json(folder / "report_reference.json")
    template = _read_json(folder / "report_template.json")

    result = compare_report.compare(report, reference, template)
    (folder / "output" / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if not result["original_available"]:
        log.warning("compare-report %s: no report_reference.json - run "
                   "extract_from_final_report.py to compare against the original", case_id)
    return result


@app.get("/case/{case_id}", tags=["cases"],
         summary="What evidence a case has, and how far the pipeline has got",
         responses=_example(
             {"case": "altamira",
              "evidence_files": {"photos": ["01.jpg", "02.jpg", "03.jpg", "04.jpg"],
                                 "audios": ["01.mp3", "02.mp3"],
                                 "field_notes": True, "occurrence_summary": True,
                                 "blank_forms": ["1-finding-notice.jpg"],
                                 "final_report_photo": True},
              "pipeline_status": {"evidence_built": True, "fact_count": 46,
                                  "evidence_summary": {"observed": 40, "inferred": 1,
                                                       "missing": 5},
                                  "report_generated": True,
                                  "report_summary": {"supported": 4, "partial": 2},
                                  "original_report_extracted": True,
                                  "comparison_available": True}},
             "Everything a UI needs for a case overview in one call."))
def get_case(case_id: str):
    """Everything the UI needs for the case overview: what evidence exists,
    and how far the pipeline has progressed for it."""
    folder = _case_dir(case_id)

    def _list(sub, exts=None):
        d = folder / sub
        if not d.is_dir():
            return []
        return sorted(p.name for p in d.iterdir()
                      if p.is_file() and (exts is None or p.suffix.lower() in exts))

    evidence = _read_json(folder / "output" / "evidence.json")
    report = _read_json(folder / "output" / "report.json")
    comparison = _read_json(folder / "output" / "comparison.json")
    reference = _read_json(folder / "report_reference.json")

    return {
        "case": case_id,
        "evidence_files": {
            "photos": _list("photos", {".jpg", ".jpeg", ".png"}),
            "audios": _list("audios", {".mp3", ".m4a", ".wav", ".ogg"}),
            "field_notes": (folder / "field-notes.md").exists(),
            "occurrence_summary": (folder / "occurrence-summary.txt").exists(),
            "blank_forms": _list("blank-forms", {".jpg", ".jpeg", ".png"}),
            "final_report_photo": (folder / "final_report.jpg").exists(),
        },
        "pipeline_status": {
            "evidence_built": evidence is not None,
            "fact_count": len(evidence["facts"]) if evidence else 0,
            "evidence_summary": evidence["summary"] if evidence else {},
            "report_generated": report is not None,
            "report_summary": report["summary"] if report else {},
            "original_report_extracted": reference is not None,
            "comparison_available": comparison is not None,
        },
    }


@app.get("/case/{case_id}/traceability", tags=["report"],
         summary="Every sentence of the report, with the facts it cites",
         responses=_example(
             {"case": "altamira",
              "sections": [{**_SECTION_EXAMPLE,
                            "clauses": [{**_SECTION_EXAMPLE["clauses"][0],
                                         "facts": [_FACT_EXAMPLE]}]}]},
             "Each clause with the full fact object(s) behind it, so \"this sentence came "
             "from exactly these places\" needs no second lookup."))
def get_traceability(case_id: str):
    """Claim -> evidence, for every section of the generated report.

    Each clause is returned together with the full fact object(s) it cites
    (value, status, confidence, and every source: file, region/segment/line)
    so a UI can render "this sentence came from exactly these places" without
    a second lookup.
    """
    folder = _case_dir(case_id)
    report = _require_json(folder / "output" / "report.json",
                           f"POST /generate-report?case_id={case_id}")
    evidence = _require_json(folder / "output" / "evidence.json",
                             f"POST /process-case?case_id={case_id}")
    facts_by_id = {f["id"]: f for f in evidence["facts"]}

    sections = []
    for section in report["sections"]:
        clauses = []
        for clause in section.get("clauses", []):
            facts = [facts_by_id[fid] for fid in clause.get("fact_ids", []) if fid in facts_by_id]
            clauses.append({**clause, "facts": facts})
        sections.append({**{k: v for k, v in section.items() if k != "clauses"}, "clauses": clauses})

    return {"case": case_id, "sections": sections}


@app.get("/case/{case_id}/evidence/{relpath:path}", tags=["files"],
         summary="Serve one raw evidence file",
         response_class=FileResponse,
         responses={200: {"description": "The file itself - a photo, a voice note, a form.",
                          "content": {"image/jpeg": {}, "audio/mpeg": {}, "text/plain": {}}}})
def get_evidence_file(case_id: str, relpath: str):
    """Serve one raw evidence file (a photo, an audio note, a form) so a
    traceability view can show the primary source next to the claim it
    backs, not just its filename."""
    folder = _case_dir(case_id)
    target = (folder / relpath).resolve()

    if folder.resolve() not in target.parents or target.suffix.lower() not in _EVIDENCE_EXTENSIONS:
        raise HTTPException(400, "not a servable evidence path")
    if not target.is_file():
        raise HTTPException(404, f"no such file: {relpath!r}")

    return FileResponse(target)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
