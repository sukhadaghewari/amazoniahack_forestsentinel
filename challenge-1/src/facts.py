"""
The shape every extracted fact takes, and where its cache file lives.

One fact is one claim about one field, carrying where it came from and how
far the pipeline trusts it:

    {"value": "SEMMA - ALTAMIRA", "source": "photos/01.jpg#stamp",
     "confidence": 0.91, "raw": "SEMMA - ALTAMIRA"}

value is None when the evidence doesn't support one - present in the output
but empty, never absent and never guessed, because a missing fact is
recoverable and a confidently wrong one is not. raw keeps what was actually
written or read, so a report can always show the source wording rather than
the pipeline's reading of it.

This lives in its own module rather than in whichever extractor needed it
first because the later cross-evidence step has to compare facts drawn from
photos, audio and text against each other, and that only works while all
three agree on the shape.
"""

import re
import unicodedata
from pathlib import Path


# Small local models emit their reasoning even when asked not to, and ollama
# passes it through in message.content - sometimes with only the closing tag,
# so there is nothing to pair it with. Everything up to and including the last
# closing tag is reasoning, never the answer. This lives here because every
# caller of a local model has to do it: an unstripped stream once shipped a
# model's entire chain of thought, prompt rules included, into a filed report.
_THINK_CLOSE_RE = re.compile(r"(?s)^.*</think\s*>")
_THINK_ANY_RE = re.compile(r"(?s)<\s*/?\s*think\s*>")


def strip_reasoning(text: str | None) -> str:
    """Whatever the model said after its last </think>, stripped.

    A no-op on output that carries no reasoning markers, so it is safe to call
    on every model response - including schema-constrained JSON, where leaked
    reasoning would otherwise turn into a parse error instead of an answer.
    """
    return _THINK_CLOSE_RE.sub("", text or "").strip()


def has_reasoning_markers(text: str) -> bool:
    """A leftover <think> or </think> anywhere - output that cannot be trusted."""
    return bool(_THINK_ANY_RE.search(text or ""))


def normalize(text) -> str:
    """Lowercased, unaccented, single-spaced - the form every comparison in the
    pipeline uses, so "Travessão" and "travessao" are never two things."""
    stripped = "".join(c for c in unicodedata.normalize("NFKD", str(text))
                       if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", stripped.lower()).strip()


def value_in_text(text, value) -> bool:
    """Is this value literally in that passage?

    The test a citation has to pass to be worth printing beside a value. A
    comma and a dot decimal are the same number, and case and accents are
    ignored; nothing else is inferred. A coordinate stamped 3°18'13"S does not
    "contain" -3.30361, and saying it did would be the pipeline marking its own
    homework.
    """
    if text is None or value is None:
        return False
    needle = normalize(value).replace(",", ".")
    return bool(needle) and needle in normalize(text).replace(",", ".")


# A word this long in a value is distinctive enough to look for on its own;
# shorter ones ("de", "ha", "3", "20") say nothing about which passage is meant.
_DISTINCTIVE_LENGTH = 4


def value_supported_by(text, value) -> bool:
    """Is that passage *about* this value, rather than merely cited next to it?

    Looser than value_in_text by one step, for the cases that need it: a value
    is often a tidied-up reading of what the source actually says. "00318,
    barra, 2026" dictated into a phone is the document number 00318/2026, and
    "igarape santa rosa" typed without accents is Igarapé Santa Rosa. So every
    distinctive word of the value must appear in the passage, rather than the
    value appearing whole.

    Every one, not any one: requiring all of them is what separates the real
    citation from a plausible neighbour. "2026" alone would let any date in the
    day's notes stand in for a document number, while demanding both "00318"
    and "2026" does not. A value whose words are all short - a bare number like
    23,418 - has to appear literally, because no rewording makes a passage
    about 412,5 ha into evidence for it.
    """
    if value_in_text(text, value):
        return True

    words = [w for w in re.split(r"[^0-9a-z]+", normalize(value).replace(",", "."))
             if len(w) >= _DISTINCTIVE_LENGTH]
    if not words:
        return False

    passage = normalize(text).replace(",", ".")
    return all(word in passage for word in words)


def cites_value(source: dict, value) -> bool:
    """value_in_text against one source's own quoted passage."""
    return value_in_text(source.get("text"), value)


def relative_file(source: dict) -> str:
    """The source's file as a reader should see it.

    Extractors cite the same file both bare and prefixed with data/<case>/, and
    voice notes by bare filename; neither is wrong, but a citation list mixing
    the two reads as two different files.
    """
    name = re.sub(r"^(?:.*/)?data/[^/]+/", "", source.get("file", ""))
    if source.get("modality") == "audio" and "/" not in name:
        return f"audios/{name}"
    return name


def citing_sources(fact: dict) -> list[dict]:
    """The sources worth showing for one fact: those whose own text contains the
    value, or all of them when none does.

    An extractor credits every passage it looked at, not only the one the value
    came from - so five field-note lines can end up cited for one area figure,
    four of them about something else entirely. Showing all five next to the
    number reads as grounding when it is not. Where no quote contains the value
    (a coordinate read off a stamp in degrees, a bare number in a form field)
    every source is shown, because then the file and locator are the citation.
    """
    sources = fact.get("sources", [])
    return [s for s in sources if cites_value(s, fact.get("value"))] or sources


def fact(value=None, source=None, confidence=0.0, raw=None, citations=None) -> dict:
    """One extracted claim - see the module docstring for what each field
    means and why value is allowed to be None.

    citations, where an extractor can cite more than one passage, is the
    authoritative pairing: [{"source": id, "quote": text}, ...]. source and raw
    stay as joined strings so the cache still reads on its own, but they must
    never be split apart and zipped back together - one is a set of ids and the
    other a list of passages, and re-pairing them by position put every field
    note's quote against the wrong timestamp.
    """
    entry = {"value": value, "source": source, "confidence": round(confidence, 3), "raw": raw}
    if citations is not None:
        entry["citations"] = citations
    return entry


def missing() -> dict:
    """A field the evidence didn't support: present, empty, zero confidence."""
    return fact()


def cache_path(folder: Path, name: str, tier: str) -> Path:
    """<folder>/.cache/<name>_<tier>.json, with the directory created.

    The tier belongs in the filename: the same photo or voice note read by
    the fast on-device model and by the heavier one are two different
    readings, and neither should quietly stand in for the other when the
    tier changes between runs.
    """
    path = Path(folder) / ".cache" / f"{name}_{tier}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
