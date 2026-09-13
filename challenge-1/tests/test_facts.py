"""
Tests for facts.py - the grounding primitives every view depends on.

These are what stop a value from being printed under a quote that says nothing
about it, and what stop a model's reasoning from being mistaken for its answer.
"""

import facts


def _source(text, **extra):
    return {"modality": "field_note", "file": "field-notes.md", "text": text, **extra}


def test_a_quote_counts_only_when_the_value_is_in_it():
    assert facts.cites_value(_source("operador valdenir moraes cardoso, diz q é so tomador"),
                             "Valdenir Moraes Cardoso")
    assert not facts.cites_value(_source("faixa queimada na porcao norte"), "243.58")


def test_decimal_comma_and_dot_are_the_same_number():
    """The field notes write 412,5 and the extractor stores 412.5."""
    assert facts.cites_value(_source("imovel 412,5 ha / RL 243,58"), "412.5")
    assert facts.cites_value(_source("area de 23.4 ha"), "23,4")


def test_a_coordinate_in_degrees_does_not_contain_its_decimal_form():
    """The check never infers: it reports what is literally there, so the view
    can show the stamp as a citation without claiming the number is in it."""
    assert not facts.cites_value(_source("3°18'13\"S 52°23'00\"0"), "-3.30361")


def test_only_the_sources_that_show_the_value_are_cited():
    """An extractor credits every passage it read. Four field-note lines about
    a tractor, a CAR number and a creek were cited for one area figure; showing
    all five beside the number reads as grounding and is the opposite of it."""
    fact = {"value": "412.5", "sources": [
        _source("imovel 412,5 ha / RL 243,58", location="08:34"),
        _source("trator komatsu D51 parado no meio da area", location="09:15"),
        _source("CAR PA-1500602 situacao ATIVO", location="09:52"),
    ]}

    cited = facts.citing_sources(fact)

    assert len(cited) == 1
    assert cited[0]["location"] == "08:34"


def test_every_source_is_kept_when_none_quotes_the_value():
    """Then the file and the locator are the citation, and the view shows them
    without a quote rather than hiding where the value came from."""
    fact = {"value": [-3.30361, -52.38333], "sources": [
        _source("Trator de esteira utilizado no desmate"),
        {"modality": "photo_stamp", "file": "photos/01.jpg", "region": "stamp",
         "text": "3°18'13\"S 52°23'00\"0"},
    ]}

    assert len(facts.citing_sources(fact)) == 2


def test_file_citations_read_the_same_however_the_extractor_wrote_them():
    assert facts.relative_file({"file": "data/altamira/occurrence-summary.txt"}) \
        == facts.relative_file({"file": "occurrence-summary.txt"})
    assert facts.relative_file({"file": "photos/01.jpg"}) == "photos/01.jpg"
    assert facts.relative_file({"file": "04.mp3", "modality": "audio"}) == "audios/04.mp3"


def test_a_value_needs_every_distinctive_word_of_it_in_the_passage():
    """One word is not enough: "2026" alone would let any date in the day's
    notes stand in as the citation for document 00318/2026."""
    spoken = "Lavrado auto de constatação número 00318, barra, 2026, 00318 de 2026."
    assert facts.value_supported_by(spoken, "00318/2026")
    assert not facts.value_supported_by("12 de março de 2026, sete e cinquenta e dois",
                                        "00318/2026")


def test_a_bare_number_has_to_appear_literally():
    """No rewording makes a passage about 412,5 ha into evidence for 23,418."""
    assert facts.value_supported_by("Área total suprimida 23,418 hectares", "23.418")
    assert not facts.value_supported_by("Imóvel de 412,5 hectares, reserva legal 243,58",
                                        "23.418")


def test_a_tidied_up_reading_still_matches_what_the_source_wrote():
    assert facts.value_supported_by("igarape santa rosa: derrubada entra na APP",
                                    "Igarapé Santa Rosa")
    assert not facts.value_supported_by("Tô com a Clay de Ano dirigindo", "Cleidiane")


def test_model_reasoning_is_never_mistaken_for_an_answer():
    """The bug this guards against: ollama streams a model's chain of thought
    into message.content, sometimes with only the closing tag, and a whole
    internal monologue - prompt rules included - was once filed as section 8
    of a report. Both extractors run their answers through this."""
    leaked = (
        "Okay, I need to read these sentences. The user says: use apenas as "
        "informações presentes. Let me check the date 12/03/2026.\n</think>\n\n"
        "A ocorrência foi classificada como Desmatamento."
    )

    assert facts.strip_reasoning(leaked) == "A ocorrência foi classificada como Desmatamento."
    assert "</think>" not in facts.strip_reasoning(leaked)
    assert facts.strip_reasoning("<think>hmm</think>Texto final.") == "Texto final."
    assert facts.strip_reasoning("Texto sem raciocínio.") == "Texto sem raciocínio."
    assert facts.has_reasoning_markers("</think> resposta")
    assert not facts.has_reasoning_markers("resposta limpa")
