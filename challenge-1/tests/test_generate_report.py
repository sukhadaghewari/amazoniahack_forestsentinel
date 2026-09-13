"""
Tests for generate_report.py, focused on the correctness guarantees the
whole project stands or falls on: a field with no evidence must read as
explicitly missing, a field with disagreeing evidence must show both
readings, and every clause claiming a value must be traceable to a real
fact whose value actually appears in what was printed.

These build a small synthetic evidence.json rather than depending on the
Altamira sample data, so they stay fast and keep testing the code instead
of one specific case.
"""

import json

import pytest

import generate_report as gr


def _fact(id_, key, type_, value, status="observed", confidence=0.9, sources=None, **extra):
    fact = {
        "id": id_, "key": key, "type": type_, "value": value,
        "status": status, "confidence": confidence,
        "sources": sources or [{"modality": "field_note", "file": "field-notes.md", "location": "08:00"}],
    }
    fact.update(extra)
    return fact


def _write_evidence(folder, facts):
    out = folder / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "evidence.json").write_text(
        json.dumps({"case": folder.name, "facts": facts, "summary": {}}, ensure_ascii=False),
        encoding="utf-8",
    )


@pytest.fixture
def case_dir(tmp_path):
    return tmp_path / "test-case"


def test_missing_fact_is_explicit_not_invented(case_dir):
    """category is never provided: the report must say so, never guess a value."""
    _write_evidence(case_dir, [
        _fact("fact_001", "opened_at", "timestamp", "2026-03-12T08:26:00"),
    ])

    report = gr.generate(case_dir)
    objetivo = next(s for s in report["sections"] if s["id"] == "objetivo")

    assert objetivo["status"] == "missing"
    assert "Não informado" in objetivo["text"]
    assert "Desmatamento" not in objetivo["text"]  # nothing invented from thin air
    assert objetivo["clauses"][0]["fact_ids"] == []


def test_observed_fact_is_quoted_and_traceable(case_dir):
    _write_evidence(case_dir, [
        _fact("fact_001", "category", "classification", "Desmatamento"),
    ])

    report = gr.generate(case_dir)
    objetivo = next(s for s in report["sections"] if s["id"] == "objetivo")

    assert objetivo["status"] == "supported"
    assert "Desmatamento" in objetivo["text"]
    assert objetivo["clauses"][0]["fact_ids"] == ["fact_001"]
    assert report["validation"]["ok"], report["validation"]["issues"]


def test_conflicting_fact_shows_both_readings_not_one(case_dir):
    """Two sources disagree on car_registry: the report must not silently pick one."""
    fact = _fact(
        "fact_001", "car_registry", "registry", "PA-AAA", status="conflicting", confidence=0.5,
        variants=[
            {"value": "PA-AAA", "confidence": 0.6, "sources": []},
            {"value": "PA-BBB", "confidence": 0.5, "sources": []},
        ],
    )
    # car_registry has no slot of its own outside "metodologia" - route it there.
    _write_evidence(case_dir, [fact])

    report = gr.generate(case_dir)
    metodologia = next(s for s in report["sections"] if s["id"] == "metodologia")
    clause = next(c for c in metodologia["clauses"] if c["key"] == "car_registry")

    assert clause["status"] == "conflict"
    assert "PA-AAA" in clause["text"] and "PA-BBB" in clause["text"]
    assert "divergem" in clause["text"]


def test_every_supported_clause_has_source_backed_text(case_dir):
    """Cross-check across every field kind at once: nothing supported may be
    printed without the exact value of the fact_id(s) it cites."""
    _write_evidence(case_dir, [
        _fact("fact_001", "category", "classification", "Desmatamento"),
        _fact("fact_002", "opened_at", "timestamp", "2026-03-12T08:26:00"),
        _fact("fact_003", "title", "location", "Travessão da 27, km 14"),
        _fact("fact_004", "operator_name", "person", "Valdenir Moraes Cardoso"),
        _fact("fact_005", "equipment_trator", "equipment", "Komatsu D51"),
        _fact("fact_006", "area_estimate_ha", "area", 23.4),
    ])

    report = gr.generate(case_dir)
    assert report["validation"]["ok"], report["validation"]["issues"]

    cited_ids = {fid for s in report["sections"] for c in s["clauses"] for fid in c["fact_ids"]}
    assert cited_ids == {"fact_001", "fact_002", "fact_003", "fact_004", "fact_005", "fact_006"}


def test_validate_report_catches_a_clause_that_lies_about_its_own_fact():
    """If a clause's text doesn't actually contain its cited fact's value,
    validation must catch it - the backstop for a future bug in a template
    or a builder, see validate_report."""
    fact = _fact("fact_001", "category", "classification", "Desmatamento")
    report = {
        "sections": [{
            "id": "objetivo", "clauses": [{
                "key": "category", "status": "supported",
                "text": "Fiscalização referente a outra coisa qualquer.",
                "fact_ids": ["fact_001"],
            }],
        }],
        "_evidence_facts": [fact],
    }

    result = gr.validate_report(report)
    assert not result["ok"]
    assert any("category" in issue for issue in result["issues"])


def test_expected_missing_slot_is_annotated_not_flagged_as_pipeline_failure(case_dir):
    """horario is a documented, structural gap (see REPORT_SLOTS) - it should
    still read as missing, but with a note explaining why, not silently."""
    _write_evidence(case_dir, [])

    report = gr.generate(case_dir)
    horario = next(s for s in report["sections"] if s["id"] == "horario")

    assert horario["status"] == "missing"
    assert horario["expected_missing"] is True


def test_render_text_includes_every_section(case_dir):
    _write_evidence(case_dir, [_fact("fact_001", "category", "classification", "Desmatamento")])
    report = gr.generate(case_dir)
    text = gr.render_text(report)

    for section in report["sections"]:
        assert section["label"] in text


def test_english_view_is_rendered_from_the_same_facts(case_dir):
    """The language switch re-renders from facts - it never translates text, so
    no value can drift between the two languages."""
    _write_evidence(case_dir, [
        _fact("fact_001", "category", "classification", "Desmatamento"),
        _fact("fact_002", "area_estimate_ha", "area", 23.4),
    ])

    report = gr.generate(case_dir)
    objetivo = next(s for s in report["sections"] if s["id"] == "objetivo")
    metodologia = next(s for s in report["sections"] if s["id"] == "metodologia")

    assert "Desmatamento" in objetivo["text_en"]          # the value is never translated
    assert "Environmental inspection" in objetivo["text_en"]
    assert "23,4 ha" in metodologia["text"]               # pt-BR decimal comma
    assert "23.4 ha" in metodologia["text_en"]            # English decimal point
    assert gr.MISSING_TEXT["en"] in metodologia["text_en"]
    assert "Não informado" not in metodologia["text_en"]

    for section in report["sections"]:
        assert section["text"] and section["text_en"]


def test_every_section_lists_the_files_its_claims_rest_on(case_dir):
    _write_evidence(case_dir, [
        _fact("fact_001", "category", "classification", "Desmatamento",
              sources=[{"modality": "occurrence_summary",
                        "file": "data/altamira/occurrence-summary.txt"}]),
    ])

    report = gr.generate(case_dir)
    objetivo = next(s for s in report["sections"] if s["id"] == "objetivo")
    autor = next(s for s in report["sections"] if s["id"] == "autor")

    # the data/<case>/ prefix some extractors emit is stripped for display
    assert objetivo["sources"] == ["occurrence-summary.txt"]
    assert autor["sources"] == []  # nothing supports it, so nothing is cited


def test_team_section_does_not_claim_full_names_it_does_not_have(case_dir):
    """The form asks for full names; the evidence has first names as heard. It
    must state both rather than passing one off as the other."""
    _write_evidence(case_dir, [
        _fact("fact_001", "team_member", "person", "Cleidiane"),
        _fact("fact_002", "team_member", "person", "Wanderson"),
    ])

    equipe = next(s for s in gr.generate(case_dir)["sections"] if s["id"] == "equipe")

    assert equipe["status"] == "partial"
    assert "Cleidiane" in equipe["text"] and "Wanderson" in equipe["text"]
    assert "não informados nos elementos de evidência disponíveis" in equipe["text"]
    assert "Full names are not stated" in equipe["text_en"]


def test_equipment_brand_reads_as_a_brand(case_dir):
    """Typed one-handed in the field it arrives as "komatsu D51"; a filed report
    capitalises it, without touching the recorded value."""
    _write_evidence(case_dir, [_fact("fact_001", "equipment_trator", "equipment", "komatsu D51")])

    report = gr.generate(case_dir)
    metodologia = next(s for s in report["sections"] if s["id"] == "metodologia")

    assert "um trator Komatsu D51" in metodologia["text"]
    assert "A Komatsu D51 tractor" in metodologia["text_en"]
    assert report["validation"]["ok"], report["validation"]["issues"]


def test_section_confidence_is_the_weakest_claim_in_it(case_dir):
    """A paragraph is only as sound as its shakiest sentence: the section must
    report the lowest confidence behind it, not an average that hides it."""
    _write_evidence(case_dir, [
        _fact("fact_001", "category", "classification", "Desmatamento", confidence=0.99),
        _fact("fact_002", "total_suppressed_area_ha", "area", "23,418", confidence=0.62),
    ])

    report = gr.generate(case_dir)
    metodologia = next(s for s in report["sections"] if s["id"] == "metodologia")
    objetivo = next(s for s in report["sections"] if s["id"] == "objetivo")
    solicitante = next(s for s in report["sections"] if s["id"] == "solicitante")

    assert metodologia["confidence"] == 0.62
    assert objetivo["confidence"] == 0.99
    assert solicitante["confidence"] is None      # boilerplate rests on nothing
