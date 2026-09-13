"""
Minimal demo UI: the report the field evidence supports, side by side with the
one the municipality filed, and the evidence behind every line of it.

Two tabs and a language switch. The evidence is there from the start; the
report tab waits for one button, which calls generate_report.generate - the
same function the CLI and src/api.py call, not a cached blob - and reports how
long it really took. So there is no server to start and nothing here can
compute a value of its own. No model runs: drafting is deterministic and the
extractors have already written data/<case>/.cache/.

The language switch re-renders from the same facts rather than translating
anything, so no value can differ between the Portuguese and English views. The
Portuguese wording is the one of record.

    pip install streamlit
    streamlit run streamlit_app.py
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
SRC_DIR = APP_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import generate_report  # noqa: E402
from facts import cites_value, citing_sources, relative_file  # noqa: E402

st.set_page_config(page_title="Relatório de Fiscalização", page_icon="🌳", layout="wide")

# Two cosmetic rules, neither of which any layout depends on - if a later
# Streamlit renames what they select, the page renders untinted and unenlarged.
#   - The pipeline button is the one thing the demo asks anyone to click, so it
#     is sized to look like it. Nothing else on the page is a primary button.
#   - Evidence cards take a very dim tint of their confidence band (see
#     fact_card, which keys each container "fact-<band>-<id>"). Translucent on
#     purpose: no theme is configured, so the page follows the viewer's light or
#     dark setting, and a low-alpha tint reads as pastel on both where a solid
#     pastel would glare on dark.
st.markdown("""
<style>
  button[kind="primary"] { padding: 1.1rem 1rem !important; font-size: 1.15rem !important; }
  button[kind="primary"] p { font-size: 1.15rem !important; font-weight: 600 !important; }

  [class*="st-key-fact-high-"]   { background: rgba(52, 168, 83, 0.07) !important;
                                   border-color: rgba(52, 168, 83, 0.25) !important; }
  [class*="st-key-fact-medium-"] { background: rgba(234, 179, 8, 0.08) !important;
                                   border-color: rgba(234, 179, 8, 0.30) !important; }
  [class*="st-key-fact-low-"]    { background: rgba(234, 67, 53, 0.06) !important;
                                   border-color: rgba(234, 67, 53, 0.25) !important; }
</style>
""", unsafe_allow_html=True)

# (pt, en) for every piece of UI text. The report body itself is rendered by
# generate_report in both languages - none of it is translated here.
T = {
    "title": ("Relatório de Fiscalização", "Enforcement Inspection Report"),
    "tab_report": ("Relatório", "Report"),
    "tab_evidence": ("Evidências", "Evidence"),
    "generated": ("Gerado a partir das evidências de campo",
                  "Generated from the field evidence"),
    "original": ("Relatório original da SEMMA", "SEMMA's own filed report"),
    "header": ("{photos} fotos · {audios} áudios · notas de campo · registro do aplicativo "
               "→ {n} fatos, {missing} campos sem evidência",
               "{photos} photos · {audios} voice notes · field notes · app record "
               "→ {n} facts, {missing} fields with no evidence"),
    "cost": ("**Custo por ocorrência: R$ 0,00.** Nenhuma chamada a API externa, nenhum "
             "dado sai da máquina. {models}. Um caso completo leva {total} de CPU num "
             "laptop ({breakdown}) - e a redação do relatório não chama modelo nenhum: "
             "é determinística a partir dos fatos.",
             "**Cost per occurrence: R$ 0.00.** No external API call, no data leaves the "
             "machine. {models}. A complete case takes {total} of laptop CPU "
             "({breakdown}) - and drafting the report calls no model at all: it is "
             "deterministic from the facts."),
    "not_transcribed": ("nosso extrator não conseguiu ler esta seção da foto",
                        "our extractor could not read this section from the photo"),
    "see_photo": ("Ver a foto do relatório original", "See the photo of the original report"),
    "no_evidence": ("Nenhuma evidência sustenta este campo",
                    "No evidence supports this field"),
    "boilerplate": ("Texto impresso no formulário", "Printed on the form itself"),
    "draft_button": ("Rodar o pipeline neste caso",
                     "Run the pipeline on this case"),
    "draft_intro": ("O botão roda o pipeline de verdade, nesta máquina: os modelos leem "
                    "os áudios, as fotos e as notas de campo, e só depois o relatório é "
                    "montado a partir dos fatos que sobreviveram à verificação. Os "
                    "modelos leem; nenhum deles escreve o relatório.",
                    "The button runs the real pipeline, on this machine: the models read "
                    "the voice notes, the photos and the field notes, and only then is "
                    "the report assembled from the facts that survived verification. The "
                    "models read; none of them writes the report."),
    "draft_force": ("Reprocessar também os áudios e as fotos (ignora o cache, ~2 min)",
                    "Re-process the voice notes and photos too (ignores the cache, ~2 min)"),
    "draft_again": ("Rodar de novo", "Run it again"),
    "stage_cached": ("já em cache, nada a fazer", "already cached, nothing to do"),
    "stage_failed": ("Falhou: {stage}. O relatório não foi montado.",
                     "Failed: {stage}. The report was not assembled."),
    "stage_ollama": ("O Ollama não respondeu. Abra o app do Ollama, ou rode "
                     "`ollama serve`, e tente de novo.",
                     "Ollama did not answer. Open the Ollama app, or run "
                     "`ollama serve`, and try again."),
    "drafting": ("Montar o relatório a partir dos fatos",
                 "Assemble the report from the facts"),
    "drafting_model": ("sem modelo - determinístico", "no model - deterministic"),
    "draft_done": ("Pipeline concluído em {total:.1f} s: {calls} chamadas de modelo na "
                   "leitura das evidências, nenhuma na redação. {sections} seções "
                   "montadas a partir de {n} fatos em {ms:.0f} ms, cada cláusula "
                   "verificada contra o fato que cita.",
                   "Pipeline finished in {total:.1f} s: {calls} model calls reading the "
                   "evidence, none writing the report. {sections} sections assembled "
                   "from {n} facts in {ms:.0f} ms, every clause checked against the "
                   "fact it cites."),
    "no_claim": ("Não é uma afirmação sobre este caso - não precisa de evidência",
                 "Not a claim about this case - needs no evidence"),
    "source_one": ("1 fonte", "1 source"),
    "source_many": ("{n} fontes", "{n} sources"),
    "bands": {
        "high": ("alta", "high"),
        "medium": ("média", "medium"),
        "low": ("baixa", "low"),
    },
    "section_of_report": ("Seção {n} · {label}", "Section {n} · {label}"),
    "section_unnumbered": ("{label}", "{label}"),
    "unused_title": ("Outros {n} fatos extraídos, que o formulário não pede",
                     "{n} other extracted facts the form does not ask for"),
    "unused_hint": ("Metadado das fotos: carimbo de GPS, horário e órgão. Extraído e "
                    "auditável, mas nenhuma seção do relatório o solicita.",
                    "Photo metadata: GPS stamp, time and agency. Extracted and auditable, "
                    "but no section of the report asks for it."),
    "derived": ("calculado", "calculated"),
    "points": ("{n} pontos GPS, a partir de {first}", "{n} GPS fixes, from {first}"),
    "api": ("API e documentação interativa: {url}", "API and interactive docs: {url}"),
    "no_evidence_file": (
        "Nenhum `output/evidence.json` em `data/{case}/`. Gere-o com: "
        "`python src/build_evidence.py data/{case}`",
        "No `output/evidence.json` in `data/{case}/`. Build it with: "
        "`python src/build_evidence.py data/{case}`"),
}

# What each kind of source is, in one glance. The icon carries most of it.
MODALITIES = {
    "occurrence_summary": ("📄", ("registro do aplicativo", "app record")),
    "exif": ("📷", ("metadado da foto", "photo metadata")),
    "photo_stamp": ("📷", ("carimbo na foto", "stamp on the photo")),
    "field_note": ("📝", ("nota digitada", "typed note")),
    "llm_extraction": ("📝", ("texto de campo", "field text")),
    "audio": ("🎙️", ("nota de voz", "voice note")),
}

# Confidence bands. The thresholds sit just under the scorer's own numbers for
# a stamp read by OCR (0.85) and a typed note (0.80), so a band change always
# means the kind of source changed.
BANDS = ((0.85, "high"), (0.70, "medium"), (0.0, "low"))
DOTS = {"high": "●●●", "medium": "●●○", "low": "●○○"}

PHOTO_KEY_RE = re.compile(r"^photo_(?P<file>.+\.(?:jpg|jpeg|png))_(?P<attr>\w+)$")

PHOTO_ATTRS = {
    "caption": ("Legenda no aplicativo", "Caption in the app"),
    "coordinate": ("Coordenada", "Coordinate"),
    "timestamp": ("Data e hora", "Date and time"),
    "agency": ("Órgão no carimbo", "Agency on the stamp"),
}

LOCATORS = {
    "field": ("campo “{v}”", "field “{v}”"),
    "region": ("carimbo", "stamp"),
    "location": ("{v}", "{v}"),
}

MODELS = ("Três modelos abertos rodando offline: faster-whisper small (voz), "
          "PP-OCRv5 (carimbos das fotos) e qwen3-vl:4b via Ollama (texto de campo)",
          "Three open models running offline: faster-whisper small (speech), PP-OCRv5 "
          "(photo stamps) and qwen3-vl:4b via Ollama (field text)")

# Measured end to end on this case (10 voice notes, 4 photos, 6 note blocks) on an
# Apple M1 Pro / 16 GB, CPU only, weights already on disk. Re-measure with
# `time python <script> data/<case>` if the pipeline changes.
RUNTIME_TOTAL = ("cerca de 2 minutos", "about 2 minutes")
RUNTIME_BREAKDOWN = ("35 s de transcrição, 32 s de OCR, 43 s de extração, 1 s de redação",
                     "35 s transcription, 32 s OCR, 43 s extraction, 1 s drafting")


def lang() -> str:
    return st.session_state.get("lang", "pt")


def pick(pair) -> str:
    return pair[0] if lang() == "pt" else pair[1]


def t(key: str, **args) -> str:
    return pick(T[key]).format(**args)


def band(confidence: float | None) -> str | None:
    if confidence is None:
        return None
    return next(name for threshold, name in BANDS if confidence >= threshold)


def show_confidence(confidence: float | None) -> str:
    name = band(confidence)
    if name is None:
        return ""
    return f"{DOTS[name]} {pick(T['bands'][name])} {confidence:.0%}"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_evidence(case: str, mtime: float):
    """The facts the extractors wrote, keyed on evidence.json's mtime: rebuild
    the evidence and the UI follows on the next rerun, without re-reading on
    every widget click."""
    return read_json(DATA_DIR / case / "output" / "evidence.json")


def count(folder: Path, sub: str, suffixes) -> int:
    d = folder / sub
    return sum(1 for p in d.glob("*") if p.suffix.lower() in suffixes) if d.is_dir() else 0


# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------

cases = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir()) if DATA_DIR.is_dir() else []
if not cases:
    st.error(f"No case folder in {DATA_DIR}")
    st.stop()

head, switch = st.columns([6, 1])
head.title(t("title"))
if switch.button("EN" if lang() == "pt" else "PT", width="stretch"):
    st.session_state["lang"] = "en" if lang() == "pt" else "pt"
    st.rerun()

case = cases[0] if len(cases) == 1 else st.selectbox("case", cases, label_visibility="collapsed")
folder = DATA_DIR / case

evidence_path = folder / "output" / "evidence.json"
if not evidence_path.exists():
    st.warning(t("no_evidence_file", case=case))
    st.stop()

evidence = load_evidence(case, evidence_path.stat().st_mtime)
reference = read_json(folder / "report_reference.json") or {}
reference_values = reference.get("values", {})

# A draft belongs to the evidence it was made from: if either the case or
# evidence.json changes, the old draft is dropped rather than shown against
# facts it was not built on.
draft_key = (case, evidence_path.stat().st_mtime)
if st.session_state.get("draft_key") != draft_key:
    st.session_state["draft_key"] = draft_key
    st.session_state.pop("report", None)
    st.session_state.pop("draft_ms", None)
report = st.session_state.get("report")

st.caption(f"**{case}** · " + t(
    "header",
    photos=count(folder, "photos", {".jpg", ".jpeg", ".png"}),
    audios=count(folder, "audios", {".mp3", ".m4a", ".wav", ".ogg"}),
    n=len(evidence["facts"]), missing=evidence["summary"].get("missing", 0)))

tab_report, tab_evidence = st.tabs([t("tab_report"), t("tab_evidence")])


# --------------------------------------------------------------------------
# 1. the two reports, side by side, in the order of the form
# --------------------------------------------------------------------------

class Stage(NamedTuple):
    """One extractor, as the pipeline button runs it."""
    script: str
    cache: str                  # what it writes; present means it can be skipped
    label: tuple[str, str]      # (pt, en)
    model: tuple[str, str]      # what actually runs, or that nothing does
    calls: int                  # model calls when it does run, for the summary
    forceable: bool = False     # --force re-reads files it has already done


# In order. Each is the same command the CLI documents, run as a subprocess so
# no model library is ever imported into this Streamlit process: a 500 MB
# whisper load inside the web app is how a 16 GB laptop dies mid-demo.
PIPELINE = (
    Stage("extract_from_occurrence_summary.py", ".cache/occurrence_summary.json",
          ("Ler o registro do aplicativo", "Read the app's occurrence record"),
          ("sem modelo - parser de linhas", "no model - line parser"), 0),
    Stage("extract_transcriptions.py", ".cache/transcripts_standard.json",
          ("Transcrever os áudios", "Transcribe the voice notes"),
          ("faster-whisper small, int8", "faster-whisper small, int8"), 10, True),
    Stage("extract_text_from_photos.py", ".cache/photo_stamps_standard.json",
          ("Ler os carimbos das fotos", "Read the stamps burned into the photos"),
          ("PaddleOCR PP-OCRv5 server", "PaddleOCR PP-OCRv5 server"), 4, True),
    Stage("extract_from_field_notes.py", "",
          ("Extrair fatos das notas e das transcrições",
           "Extract facts from the field notes and transcripts"),
          ("qwen3-vl:4b via Ollama, 5 perguntas", "qwen3-vl:4b via Ollama, 5 questions"), 5),
    Stage("build_evidence.py", "",
          ("Reunir, corroborar e pontuar as evidências",
           "Merge, corroborate and score the evidence"),
          ("sem modelo - determinístico", "no model - deterministic"), 0),
)


def run_stage(stage: Stage, folder: Path, force: bool):
    """Run one extractor and hand back its own log lines as it writes them.

    CONSOLE_LOG_LEVEL is what makes this narrate itself - see src/logconf.py.
    Yielding rather than returning is what lets the caller show the lines while
    a 40-second model call is still going.
    """
    argv = [sys.executable, stage.script, str(folder)]
    if force and stage.forceable:
        argv.append("--force")

    env = os.environ | {"CONSOLE_LOG_LEVEL": "INFO", "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(argv, cwd=SRC_DIR, env=env, text=True, bufsize=1,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in process.stdout:
        yield line.rstrip()
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"{stage.script} exited {process.returncode}")


def run_pipeline(folder: Path, force: bool) -> tuple[bool, int, float]:
    """Every stage, each in its own status box.

    Returns (all stages succeeded, model calls made, seconds). A failed stage
    stops the run: drafting against half-written evidence would produce a
    report that looks fine and is missing whatever that stage reads.
    """
    started = time.perf_counter()
    calls = 0

    for stage in PIPELINE:
        cached = bool(stage.cache) and (folder / stage.cache).exists() and not force
        title = f"{pick(stage.label)} — {pick(stage.model)}"
        with st.status(title, expanded=not cached) as box:
            if cached:
                box.update(label=f"{title} · {t('stage_cached')}", state="complete")
                continue

            lines: list[str] = []
            output = st.empty()
            stage_started = time.perf_counter()
            try:
                for line in run_stage(stage, folder, force):
                    lines.append(line)
                    output.code("\n".join(lines[-12:]), language="log")
            except (RuntimeError, OSError) as exc:
                box.update(label=f"{title} · {exc}", state="error")
                st.error(t("stage_failed", stage=stage.script))
                if "field_notes" in stage.script or "final_report" in stage.script:
                    st.warning(t("stage_ollama"))
                return False, calls, time.perf_counter() - started

            calls += stage.calls
            box.update(state="complete", expanded=False,
                       label=f"{title} · {time.perf_counter() - stage_started:.1f} s")

    return True, calls, time.perf_counter() - started


def heading(slot: dict) -> str:
    """Works on a REPORT_SLOTS entry or on a drafted section - both carry the
    number and both labels, so the evidence tab needs no draft to head its
    groups."""
    label = slot["label"] if lang() == "pt" else slot["label_en"]
    key = "section_of_report" if slot["number"] else "section_unnumbered"
    return t(key, n=slot["number"], label=label)


with tab_report:
    if report is None:
        st.info(t("draft_intro"))
        force = st.checkbox(t("draft_force"), key="force_models")
        st.write("")
        _, middle, _ = st.columns([1, 2, 1])
        if middle.button(t("draft_button"), type="primary", width="stretch"):
            ok, calls, seconds = run_pipeline(folder, force)
            if not ok:
                st.stop()       # leave the failed stage on screen, draft nothing

            with st.status(f"{t('drafting')} — {t('drafting_model')}") as box:
                started = time.perf_counter()
                # the same call the CLI and the API make, on the evidence the
                # stages above just wrote - not a cached blob
                drafted = generate_report.generate(folder)
                ms = (time.perf_counter() - started) * 1000
                box.update(state="complete", label=f"{t('drafting')} · {ms:.0f} ms")

            st.session_state["report"] = drafted
            st.session_state["draft_ms"] = ms
            st.session_state["pipeline_calls"] = calls
            st.session_state["pipeline_seconds"] = seconds + ms / 1000
            # the stages rewrote evidence.json, so re-key the cache to its new mtime
            st.session_state["draft_key"] = (case, evidence_path.stat().st_mtime)
            st.rerun()
    else:
        if not report["validation"]["ok"]:
            st.error("\n".join(report["validation"]["issues"]))

        done, again = st.columns([6, 1])
        done.success(t("draft_done",
                       total=st.session_state.get("pipeline_seconds", 0.0),
                       calls=st.session_state.get("pipeline_calls", 0),
                       sections=len(report["sections"]),
                       n=len(evidence["facts"]),
                       ms=st.session_state.get("draft_ms", 0.0)))
        if again.button(t("draft_again"), width="stretch"):
            st.session_state.pop("report", None)
            st.rerun()

        left, right = st.columns(2)
        left.markdown(f"##### {t('generated')}")
        right.markdown(f"##### {t('original')}")

        for section in report["sections"]:
            st.markdown(f"**{heading(section)}**")
            left, right = st.columns(2)

            left.write(section["text"] if lang() == "pt" else section["text_en"])
            if section["status"] == "static":
                left.caption(t("boilerplate"))
            elif section["status"] == "missing":
                left.caption(t("no_evidence"))
            else:
                files = section["sources"]
                mark = t("source_one") if len(files) == 1 else t("source_many", n=len(files))
                left.caption(f"{show_confidence(section['confidence'])} · {mark}: "
                             + ", ".join(files))

            original = (reference_values.get(section["id"]) or "").strip()
            if original:
                right.write(original)
            elif section["status"] == "static":
                right.write(section["text"])       # the same boilerplate is on both forms
                right.caption(t("boilerplate"))
            else:
                right.caption(f"_{t('not_transcribed')}_")

            st.divider()

        photo = folder / "final_report.jpg"
        if photo.exists():
            with st.expander(t("see_photo")):
                st.image(str(photo), width=620)


# --------------------------------------------------------------------------
# 2. the evidence, grouped by the report section it answers
# --------------------------------------------------------------------------

def show_value(fact: dict) -> str:
    value, kind = fact["value"], fact["type"]
    if value is None:
        return "—"
    if kind == "coordinate":
        return f"{value[0]}, {value[1]}"
    if kind == "coordinate_set":
        return t("points", n=len(value), first=f"{value[0][0]}, {value[0][1]}")
    if kind == "timestamp":
        try:
            return datetime.fromisoformat(value).strftime("%d/%m/%Y %H:%M:%S")
        except ValueError:
            return str(value)
    formatted = generate_report.format_value(fact["key"], fact, lang())
    return f"{formatted} ha" if kind == "area" else formatted


def show_locator(source: dict) -> str:
    if "start" in source:
        seconds = int(source["start"])
        return f"{seconds // 60}:{seconds % 60:02d}"
    for field, pair in LOCATORS.items():
        if source.get(field):
            return pick(pair).format(v=source[field])
    return ""


def quote_of(source: dict, fact: dict) -> str:
    """The source's own words, but only when the value is really in them.

    Several renderings of one value count as a match, because the source and
    the report write it differently: a timestamp is ISO in evidence.json and
    "12/03/2026 08:26" in the app's export, an area has a dot in one and a
    comma in the other. What does not count is a passage that simply happens to
    be cited - an extractor credits every passage it read, and a number printed
    under a quote about something else reads as grounding while being the
    opposite of it.
    """
    if source.get("modality") == "photo_stamp":
        # OCR of one region of the image: these words are the source itself,
        # not a passage anything chose, so they are shown even where the value
        # is written differently - a coordinate burned in as 3°18'13"S is the
        # same reading as -3.30361, and the reader can see that for themselves.
        return str(source.get("text", "")).strip()

    candidates = [fact["value"], show_value(fact)]
    if fact["type"] == "timestamp":
        try:
            moment = datetime.fromisoformat(fact["value"])
            candidates += [moment.strftime("%d/%m/%Y %H:%M"), moment.strftime("%H:%M:%S")]
        except (TypeError, ValueError):
            pass
    if any(cites_value(source, candidate) for candidate in candidates):
        return str(source["text"]).strip()
    return ""


def fact_card(fact: dict, label: str) -> None:
    """One fact: what it says, how sure we are, and where it came from.

    Tinted by the same band() that picks its dots, so the colour can never
    disagree with the label printed on it. The key is also what makes the tint
    possible: Streamlit turns it into a CSS class. Each fact renders once per
    run (see the evidence tab's `shown` set), so the key is unique.
    """
    tint = band(fact["confidence"]) or "none"
    with st.container(border=True, key=f"fact-{tint}-{fact['id']}"):
        top, score = st.columns([3, 1])
        top.caption(label)
        top.markdown(f"### {show_value(fact)}")
        score.caption(show_confidence(fact["confidence"]))

        if fact.get("derivation"):
            # a computed value's grounding is its arithmetic, not a passage
            score.caption(f"{t('derived')}: `{fact['derivation']}`")

        for source in citing_sources(fact):
            icon, name = MODALITIES.get(source["modality"], ("•", ("fonte", "source")))
            where = " · ".join(x for x in [f"`{relative_file(source)}`",
                                           show_locator(source)] if x)
            st.caption(f"{icon} {pick(name)} — {where}")
            quote = quote_of(source, fact)
            if quote:
                st.markdown(f"> {re.sub(r'([*_`])', r'\\\1', quote)}")


def empty_card(label: str, reason: str | None = None) -> None:
    """A field the form asks for and nothing in the evidence answers."""
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"**—** {t('no_evidence').lower()}")
        if reason:
            st.caption(reason)


def static_card(slot: dict) -> None:
    """A section the blank form already answers. It has no fact behind it, but
    it still gets a card: skipping it made the evidence tab open at section 2,
    which reads as a missing section rather than as a section needing nothing.
    """
    with st.container(border=True):
        st.caption(t("boilerplate"))
        st.markdown(f"**{slot['value' if lang() == 'pt' else 'value_en']}**")
        st.caption(t("no_claim"))


def fact_label(fact: dict) -> str:
    photo = PHOTO_KEY_RE.match(fact["key"])
    if photo:
        attr = pick(PHOTO_ATTRS.get(photo["attr"], (photo["attr"], photo["attr"])))
        return f"{photo['file']} · {attr}"
    return generate_report.label(fact["key"], lang())


with tab_evidence:
    facts_by_key: dict[str, list[dict]] = {}
    for fact in evidence["facts"]:
        facts_by_key.setdefault(fact["key"], []).append(fact)

    shown = set()
    for slot in generate_report.REPORT_SLOTS:
        if slot["kind"] == "static":
            st.markdown(f"##### {heading(slot)}")
            static_card(slot)
            continue

        # A fact appears once, under the first section that draws on it. Section 8
        # restates what 3, 4 and 5 already said, and showing "Desmatamento" three
        # times with the same citation reads as a bug rather than as thoroughness.
        members = [f for key in slot["fills_from"] for f in facts_by_key.get(key, [])
                   if f["id"] not in shown]
        shown.update(f["id"] for f in members)

        st.markdown(f"##### {heading(slot)}")
        if not members:
            empty_card(generate_report.label(slot["fills_from"][0], lang()))
            continue

        columns = st.columns(2)
        for index, fact in enumerate(members):
            with columns[index % 2]:
                if fact["status"] == "missing":
                    empty_card(fact_label(fact),
                               fact.get("reason" if lang() == "pt" else "reason_en"))
                else:
                    fact_card(fact, fact_label(fact))

    rest = [f for f in evidence["facts"] if f["id"] not in shown]
    if rest:
        with st.expander(t("unused_title", n=len(rest))):
            st.caption(t("unused_hint"))
            columns = st.columns(2)
            for index, fact in enumerate(sorted(rest, key=lambda f: f["key"])):
                with columns[index % 2]:
                    if fact["status"] == "missing":
                        empty_card(fact_label(fact),
                                   fact.get("reason" if lang() == "pt" else "reason_en"))
                    else:
                        fact_card(fact, fact_label(fact))

st.caption(t("cost", models=pick(MODELS), total=pick(RUNTIME_TOTAL),
             breakdown=pick(RUNTIME_BREAKDOWN)))
st.caption(t("api", url="`http://127.0.0.1:8000/docs`"))
