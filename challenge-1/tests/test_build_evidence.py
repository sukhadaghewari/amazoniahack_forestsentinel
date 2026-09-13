"""
Tests for build_evidence.py - the step every report is written from, and the
one where a citation silently attaches to the wrong passage.
"""

import json

import pytest

import build_evidence as be


@pytest.fixture
def case(tmp_path):
    folder = tmp_path / "case"
    (folder / ".cache").mkdir(parents=True)
    (folder / "photos").mkdir()
    return folder


def _write(folder, name, data):
    (folder / ".cache" / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _by_key(evidence):
    out = {}
    for fact in evidence["facts"]:
        out.setdefault(fact["key"], []).append(fact)
    return out


def test_citations_keep_each_quote_with_the_passage_it_came_from(case):
    """The bug this guards against: source ids were a sorted set and quotes an
    unsorted list, joined into two strings and zipped back by position, so every
    field note was filed under another block's timestamp."""
    _write(case, "field_notes.json", {
        "property_area_ha": {
            "value": "412.5", "confidence": 0.82,
            "source": "field-notes.md@10:20 + 08.mp3@10.6s",
            "raw": "imovel 412,5 ha / RL 243,58; Imóvel de 412,5 hectares.",
            "citations": [
                {"source": "field-notes.md@10:20", "quote": "imovel 412,5 ha / RL 243,58"},
                {"source": "08.mp3@10.6s", "quote": "Imóvel de 412,5 hectares."},
            ],
        },
    })

    fact = _by_key(be.build(case))["property_area_ha"][0]
    cited = {(s["file"], s.get("location") or s.get("start"), s["text"]) for s in fact["sources"]}

    assert cited == {
        ("field-notes.md", "10:20", "imovel 412,5 ha / RL 243,58"),
        ("08.mp3", 10.6, "Imóvel de 412,5 hectares."),
    }


def test_a_cache_without_citations_quotes_nothing_rather_than_guessing(case):
    _write(case, "field_notes.json", {
        "property_area_ha": {
            "value": "412.5", "confidence": 0.82,
            "source": "field-notes.md@10:20 + field-notes.md@09:15",
            "raw": "imovel 412,5 ha / RL 243,58; faixa queimada na porcao norte",
        },
    })

    fact = _by_key(be.build(case))["property_area_ha"][0]

    assert len(fact["sources"]) == 2
    assert not any("text" in source for source in fact["sources"])


def test_a_repeated_key_is_a_list_not_a_disagreement(case):
    """A team has four members. Reading the second name as a source contradicting
    the first reported one conflicting fact and lost three of them."""
    _write(case, "field_notes.json", {
        "team_members": [
            {"value": "Cleidiane", "confidence": 0.68, "source": "field-notes.md@10:46",
             "raw": "testemunhas: cleidiane e deuzimar",
             "citations": [{"source": "field-notes.md@10:46",
                            "quote": "testemunhas: cleidiane e deuzimar"}]},
            {"value": "Wanderson", "confidence": 0.68, "source": "01.mp3@26.4s",
             "raw": "o Wanderson", "citations": [{"source": "01.mp3@26.4s",
                                                  "quote": "o Wanderson"}]},
        ],
    })

    members = _by_key(be.build(case))["team_member"]

    assert {m["value"] for m in members} == {"Cleidiane", "Wanderson"}
    assert all(m["status"] == "observed" for m in members)


def test_two_sources_disagreeing_about_one_value_is_still_a_conflict(case):
    _write(case, "field_notes.json", {
        "car_registry": {"value": "PA-111", "confidence": 0.68, "source": "field-notes.md@10:20",
                         "raw": "CAR PA-111",
                         "citations": [{"source": "field-notes.md@10:20", "quote": "CAR PA-111"}]},
    })
    _write(case, "occurrence_summary.json", {
        "car_registry": {"value": "PA-222", "source": "occurrence-summary.txt#Registro CAR",
                         "confidence": 0.99, "raw": "PA-222"},
        "marked_points": [], "photos": [], "audios": [], "documents": [],
    })

    fact = _by_key(be.build(case))["car_registry"][0]

    assert fact["status"] == "conflicting"
    assert {v["value"] for v in fact["variants"]} == {"PA-111", "PA-222"}


def test_one_citation_per_passage_however_many_claims_name_it(case):
    """An inferred area cites both its inputs, and "total minus APP" is one
    sentence naming both figures - it must not be cited twice."""
    citation = [{"source": "09.mp3@11.2s",
                 "quote": "Área total suprimida 23,418 hectares, sendo 2,489 em APP."}]
    _write(case, "field_notes.json", {
        "total_suppressed_area_ha": {"value": "23,418", "confidence": 0.68,
                                     "source": "09.mp3@11.2s", "raw": "x", "citations": citation},
        "app_area_ha": {"value": "2,489", "confidence": 0.68,
                        "source": "09.mp3@11.2s", "raw": "x", "citations": citation},
    })

    inferred = _by_key(be.build(case))["reserve_legal_suppressed_ha"][0]

    assert inferred["status"] == "inferred"
    assert inferred["value"] == 20.929
    assert len(inferred["sources"]) == 1
    assert "23,418" in inferred["derivation"] and "2,489" in inferred["derivation"]


def test_photos_are_bound_by_the_number_the_app_gave_them(case):
    """Position in the list is not identity: one unparseable fix line and every
    later photo would be off by one, filing photo 4's coordinates under 3."""
    (case / "photos" / "03.jpg").write_bytes(b"")
    _write(case, "occurrence_summary.json", {
        "marked_points": [], "photos": [
            {"n": 3, "timestamp": "2026-03-12T09:07:33", "coordinate": [-3.29944, -52.38056],
             "caption": "Faixa queimada", "accuracy_m": 9},
        ], "audios": [], "documents": [],
    })

    keys = _by_key(be.build(case))

    assert "photo_03.jpg_caption" in keys
    assert not any(key.startswith("photo_01") for key in keys)


def test_fields_the_report_needs_and_nothing_supports_are_carried_as_missing(case):
    facts = _by_key(be.build(case))

    for key in be._KNOWN_ABSENT:
        assert facts[key][0]["status"] == "missing"
        assert facts[key][0]["value"] is None
        assert facts[key][0]["reason"] and facts[key][0]["reason_en"]
