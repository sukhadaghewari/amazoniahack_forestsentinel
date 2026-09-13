"""
Tests for compare_report.py: does the seven-way verdict land where it should,
and does it stay honest when the original report doesn't exist yet at all
(no report_reference.json - the pipeline must say so, not fabricate one)?
"""

import compare_report as cr


def _section(id_, number, label, kind, status, text, clauses):
    return {"id": id_, "number": number, "label": label, "label_en": label,
            "kind": kind, "status": status, "text": text, "text_en": text,
            "clauses": clauses, "expected_missing": False}


def _clause(key, status, text, fact_ids):
    return {"key": key, "status": status, "text": text, "text_en": text,
            "fact_ids": fact_ids}


def _report(sections, facts=None):
    return {"case": "test-case", "sections": sections, "_evidence_facts": facts or [],
            "validation": {"ok": True, "issues": []}}


def test_both_blank_is_match_not_a_gap():
    """Blank on the paper form and declined by the pipeline: they agree."""
    section = _section("autor", 2, "Autor", "fact", "missing",
                       "Não informado nos elementos de evidência disponíveis.", [])
    report = _report([section])

    result = cr.compare(report, reference={"values": {"autor": ""}})

    assert result["slots"][0]["verdict"] == cr.MATCH
    assert result["slots"][0]["note_code"] == "both_blank"


def test_slot_the_vision_model_never_read_is_not_scored_as_agreement():
    """A slot absent from report_reference.json was not read off the photo. That
    is not the same as blank on the paper, and must never count as a MATCH -
    the bug this guards against reported agreement with a section nobody read."""
    section = _section("autor", 2, "Autor", "fact", "missing",
                       "Não informado nos elementos de evidência disponíveis.", [])
    report = _report([section])

    result = cr.compare(report, reference={"values": {"data": "12/03/2026"}})

    assert result["slots"][0]["verdict"] == cr.NOT_COMPARABLE
    assert result["slots"][0]["note_code"] == "slot_not_transcribed"
    assert result["completeness"]["comparable_sections"] == 0


def test_original_has_it_generated_does_not_is_missing_from_generated():
    section = _section("horario", 6, "Horário", "fact", "missing",
                       "Não informado nos elementos de evidência disponíveis.", [])
    report = _report([section])

    result = cr.compare(report, reference={"values": {"horario": "08h02min"}})

    assert result["slots"][0]["verdict"] == cr.MISSING_FROM_GENERATED


def test_generated_has_it_original_blank_is_not_comparable():
    section = _section("enquadramento", None, "Enquadramento legal", "fact", "supported",
                       "Enquadramento legal: Lei Municipal 3.427.",
                       [_clause("legal_citation", "supported", "Enquadramento legal: Lei Municipal 3.427.",
                               ["fact_001"])])
    report = _report([section])

    result = cr.compare(report, reference={"values": {"enquadramento": ""}})

    assert result["slots"][0]["verdict"] == cr.NOT_COMPARABLE
    assert result["slots"][0]["note_code"] == "original_blank"


def test_matching_values_are_match():
    section = _section("objetivo", 4, "Objetivo", "composite", "supported",
                       "Fiscalização ambiental referente à ocorrência classificada como: Desmatamento.",
                       [_clause("category", "supported",
                               "Fiscalização ambiental referente à ocorrência classificada como: Desmatamento.",
                               ["fact_001"])])
    report = _report([section])
    reference = {"values": {"objetivo": "Verificação de denúncia de desmatamento na área."}}

    result = cr.compare(report, reference)

    assert result["slots"][0]["verdict"] == cr.MATCH


def test_generated_value_absent_from_original_text_is_extra_in_generated():
    section = _section("objetivo", 4, "Objetivo", "composite", "supported",
                       "Fiscalização ambiental referente à ocorrência classificada como: Queimada.",
                       [_clause("category", "supported",
                               "Fiscalização ambiental referente à ocorrência classificada como: Queimada.",
                               ["fact_001"])])
    report = _report([section])
    reference = {"values": {"objetivo": "Verificação de denúncia de desmatamento."}}

    result = cr.compare(report, reference)

    assert result["slots"][0]["verdict"] == cr.EXTRA_IN_GENERATED


def test_no_reference_file_marks_original_unavailable_not_fabricated():
    section = _section("autor", 2, "Autor", "fact", "missing",
                       "Não informado nos elementos de evidência disponíveis.", [])
    report = _report([section])

    result = cr.compare(report, reference=None)

    assert result["original_available"] is False
    # generated content is still shown, but never scored against a document
    # that was never actually read - every slot must say so explicitly.
    assert result["slots"][0]["verdict"] == cr.NOT_COMPARABLE
    assert result["slots"][0]["original_text"] is None
    assert result["generic_findings"] == []  # nothing invented from an absent original


def test_unsupported_claims_from_validation_are_surfaced():
    report = _report([], facts=[])
    report["validation"] = {"ok": False, "issues": ["objetivo/category: fact fact_001 not found in its own clause text"]}

    result = cr.compare(report, reference={"values": {}})

    assert len(result["unsupported_claims"]) == 1
    assert result["unsupported_claims"][0]["verdict"] == cr.UNSUPPORTED


def test_cpf_in_original_is_always_missing_from_generated_generic_check():
    """No extractor in this pipeline has a CPF field - any CPF the original
    report contains can never be backed by evidence, for any case."""
    reference = {"values": {"metodologia": "Sr. Fulano de Tal, CPF nº 111.222.333-44, presente no local."}}
    report = _report([], facts=[])

    result = cr.compare(report, reference)

    cpf_findings = [f for f in result["generic_findings"] if f["kind"] == "cpf"]
    assert len(cpf_findings) == 1
    assert cpf_findings[0]["verdict"] == cr.MISSING_FROM_GENERATED


def test_area_figure_within_tolerance_of_a_fact_matches():
    fact = {"id": "fact_001", "key": "area_estimate_ha", "type": "area", "value": 23.4, "status": "observed"}
    reference = {"values": {"metodologia": "apurando supressão total de 23.4187 ha na propriedade."}}
    report = _report([], facts=[fact])

    result = cr.compare(report, reference)

    area_findings = [f for f in result["generic_findings"] if f["kind"] == "area_ha"]
    assert area_findings[0]["verdict"] == cr.MATCH


def test_area_figure_with_no_supporting_fact_is_missing_from_generated():
    reference = {"values": {"metodologia": "sendo 2.489 ha em Área de Preservação Permanente."}}
    report = _report([], facts=[])

    result = cr.compare(report, reference)

    area_findings = [f for f in result["generic_findings"] if f["kind"] == "area_ha"]
    assert area_findings[0]["verdict"] == cr.MISSING_FROM_GENERATED


def test_different_date_in_original_is_a_real_conflict_not_extra_or_not_comparable():
    """CONFLICT must actually be reachable: an occurrence has exactly one
    opening date, so a different one in the original report is a genuine
    disagreement, not just 'phrased differently' (EXTRA_IN_GENERATED) or
    'can't tell' (NOT_COMPARABLE)."""
    fact = {"id": "fact_005", "key": "opened_at", "type": "timestamp",
            "value": "2026-03-12T08:26:00", "status": "observed"}
    section = _section("data", 5, "Data", "fact", "supported",
                       "A ocorrência foi aberta em 12/03/2026 às 08:26.",
                       [_clause("opened_at", "supported",
                               "A ocorrência foi aberta em 12/03/2026 às 08:26.", ["fact_005"])])
    report = _report([section], facts=[fact])
    reference = {"values": {"data": "13/03/2026"}}

    result = cr.compare(report, reference)

    assert result["slots"][0]["verdict"] == cr.CONFLICT
    assert "13/03/2026" in result["slots"][0]["note"]


def test_matching_date_is_match_not_conflict():
    fact = {"id": "fact_005", "key": "opened_at", "type": "timestamp",
            "value": "2026-03-12T08:26:00", "status": "observed"}
    section = _section("data", 5, "Data", "fact", "supported",
                       "A ocorrência foi aberta em 12/03/2026 às 08:26.",
                       [_clause("opened_at", "supported",
                               "A ocorrência foi aberta em 12/03/2026 às 08:26.", ["fact_005"])])
    report = _report([section], facts=[fact])
    reference = {"values": {"data": "12/03/2026"}}

    result = cr.compare(report, reference)

    assert result["slots"][0]["verdict"] == cr.MATCH


def test_notes_are_available_in_both_languages():
    """The UI's language switch must never fall back to a Portuguese-only note."""
    section = _section("horario", 6, "Horário", "fact", "missing",
                       "Não informado nos elementos de evidência disponíveis.", [])
    report = _report([section])

    slot = cr.compare(report, reference={"values": {"horario": "08h02min"}})["slots"][0]
    pt = cr.note_text(slot["note_code"], slot["note_args"], "pt")
    en = cr.note_text(slot["note_code"], slot["note_args"], "en")

    assert pt and en and pt != en
    assert all(code in cr.NOTES for code in {"reference_absent", "slot_not_transcribed"})


def test_partial_match_is_not_reported_as_not_comparable():
    """Most values confirmed and one not found is a partial match. Filing that
    as NOT_COMPARABLE hid a good result behind "can't tell"."""
    clauses = [
        _clause("category", "supported", "A ocorrência foi classificada como Desmatamento.",
                ["fact_001"]),
        _clause("title", "supported", "Local da ação: Travessão da 27, km 14.", ["fact_002"]),
    ]
    section = _section("metodologia", 8, "Metodologia", "narrative", "supported",
                       " ".join(c["text"] for c in clauses), clauses)
    report = _report([section])
    reference = {"values": {"metodologia": "Constatado desmatamento na propriedade."}}

    slot = cr.compare(report, reference)["slots"][0]

    assert slot["verdict"] == cr.PARTIAL_MATCH
    assert "title" in slot["note"]
    assert cr.compare(report, reference)["completeness"]["partial_sections"] == 1


def test_same_area_at_different_precision_is_not_a_mismatch():
    """The app records 23.4 ha and the original writes 23.4187 ha: one
    measurement, two roundings. The per-section check and the whole-text check
    must agree about it."""
    fact = {"id": "fact_001", "key": "area_estimate_ha", "type": "area",
            "value": 23.4, "status": "observed"}
    clause = _clause("area_estimate_ha", "supported",
                     "A área estimada da ocorrência é de 23,4 ha.", ["fact_001"])
    section = _section("metodologia", 8, "Metodologia", "narrative", "supported",
                       clause["text"], [clause])
    report = _report([section], facts=[fact])
    reference = {"values": {"metodologia": "apurando supressão total de 23.4187 ha."}}

    result = cr.compare(report, reference)

    assert result["slots"][0]["verdict"] == cr.MATCH
    assert result["generic_findings"][0]["verdict"] == cr.MATCH
