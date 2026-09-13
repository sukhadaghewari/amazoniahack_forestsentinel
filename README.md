# AmazôniaHack 4.0 — Challenge 1: evidence-grounded enforcement report

Drafts a Brazilian municipal *Relatório de Fiscalização* (environmental
inspection report) from a case's raw field evidence: voice notes, stamped
photos, the inspector's typed notes and the field app's occurrence export.
Everything runs locally on CPU, with no paid API calls.

Models only extract facts, and the report is rendered from those facts by
templates. A value that is not in the evidence therefore cannot appear in the
report.

## Models

| Input | Model | Runtime | Task |
|---|---|---|---|
| Voice (`audios/*.mp3`) | faster-whisper `small` (default) / `tiny` (`--lite`) | CTranslate2, int8 | Portuguese transcription |
| Images: photos (`photos/*.jpg`) | PaddleOCR PP-OCRv5 `server` (default, ~165 MB) / `mobile` + `latin` rec (`--lite`, ~13 MB) | PaddlePaddle | OCR of the burned-in stamp (date, coordinates, agency) |
| Images: filed form (`final_report.jpg`) | Qwen3-VL 4B (`qwen3-vl:4b`, ~3.3 GB) | Ollama | Transcribe each form section, for scoring only |
| Text (`field-notes.md` + transcripts) | Qwen3-VL 4B (`qwen3-vl:4b`) | Ollama | Named-value extraction with verbatim quotes |
| Text (`occurrence-summary.txt`) | none | line parser | Fixed-format app export |

No model is used to write report text.

## Requirements

- Python 3.12 (tested on 3.12.13)
- [Ollama](https://ollama.com/download), with the daemon running
- ~4 GB disk for weights. Tested on an M1 Pro with 16 GB.

```
faster-whisper==1.2.1  ctranslate2==4.8.2  av==18.1.0
paddleocr==3.7.0       paddlepaddle==3.3.1 piexif==1.1.3
ollama==0.6.2          huggingface_hub==0.36.2
numpy==2.3.5           pillow==12.3.0
fastapi==0.141.1       uvicorn[standard]==0.52.4
streamlit==1.63.0      pytest==9.1.1
```

## Setup (once)

```bash
# 1. environment
python3.12 -m venv .venv && source .venv/bin/activate
pip install faster-whisper paddleocr paddlepaddle piexif ollama huggingface_hub \
            numpy pillow fastapi "uvicorn[standard]" streamlit pytest

# 2. start Ollama (open the app, or run in another terminal)
ollama serve

# 3. download weights (skips anything already on disk)
cd challenge-1
python src/download_audio_models.py            # whisper tiny + small  → .models/audio/
python src/download_vision_language_models.py  # PP-OCRv5 mobile + server → .models/vision/
                                               # qwen3-vl:4b → ~/.ollama/models
```

## Run

Run all commands from `challenge-1/`, in the order below. Every script takes
the case folder as its only positional argument. `data/altamira` is the
included case.

If `data/<case>/output/evidence.json` already exists, you can go straight to
**Step 6**.

| Step | Command | Needs | Writes | Time |
|---|---|---|---|---|
| 1. Transcribe voice notes | `python src/extract_transcriptions.py data/altamira` | `audios/` | `.cache/transcripts_standard.json` | 35 s |
| 2. Read photo stamps | `python src/extract_text_from_photos.py data/altamira` | `photos/` | `.cache/photo_stamps_standard.json` | 32 s |
| 3. Parse the app export | `python src/extract_from_occurrence_summary.py data/altamira` | `occurrence-summary.txt` | `.cache/occurrence_summary.json` | <1 s |
| 4. Extract facts from text | `python src/extract_from_field_notes.py data/altamira` | `field-notes.md`, **step 1**, Ollama | `.cache/field_notes.json` | 43 s |
| 5. Build evidence | `python src/build_evidence.py data/altamira` | steps 1–4 | `output/evidence.json` | <1 s |
| 6. Draft the report | `python src/generate_report.py data/altamira` | step 5 | `output/report.json`, `output/report.txt` | <1 s |
| 7. Read the filed report *(optional)* | `python src/extract_from_final_report.py data/altamira` | `final_report.jpg`, Ollama | `report_template.json`, `report_reference.json` | — |
| 8. Score against it *(optional)* | `python src/compare_report.py data/altamira` | steps 6–7 | `output/comparison.json` | <1 s |

All paths in *Needs* and *Writes* are relative to `data/<case>/`.

Flags:
- `--lite` (steps 1, 2) uses the small models and writes `*_lite.json`. Step 5
  uses the standard caches and falls back to the lite ones.
- `--force` (steps 1, 2) re-processes files that are already cached.
- `--model TAG` (steps 4, 7) uses a different Ollama model.
- `--template-only` (step 7) writes the template and skips the model.

Steps 1–2 skip files that are already cached, but step 4 calls the model on
every run. If step 5 can't find a cache file, it logs a warning and builds
from the rest.

### Demo

```bash
./run_demo.sh      # needs output/evidence.json (step 5)
```
- **Streamlit UI** at http://localhost:8501. Report tab: the generated report
  next to the filed one. Evidence tab: every fact with its source quote and
  confidence. A PT/EN toggle and a "Run the pipeline" button that runs steps
  1–6.
- **FastAPI** at http://127.0.0.1:8000/docs (port set by `API_PORT`). Reads
  caches only and never loads a model. If a required file is missing it
  returns `409` naming the command to run.

### Tests

```bash
python -m pytest   # 45 tests on synthetic evidence, no models needed
```

Every script logs to `.logs/pipeline.log`. `LOG_LEVEL` sets what the file
records and `CONSOLE_LOG_LEVEL` what is also printed to stderr.

## What the code does (pseudocode)

**1. `extract_transcriptions.py`**
```
load whisper (small, or tiny if --lite) on CPU, int8
for each mp3 in audios/ not already in the cache:
    transcribe in Portuguese, skipping silence
    for each segment:
        confidence = exp(average log-probability)
        flag it if: probably silence (no_speech > 0.6)
                 or low confidence (avg_logprob < -1.0)
                 or repetitive text (compression_ratio > 2.4)
save start, end, text, confidence and flags for every segment
```

**2. `extract_text_from_photos.py`**
```
for each photo not already in the cache:
    read timestamp + GPS from EXIF
    crop the bottom corners where the stamp is, enlarge them, run OCR
    parse the stamp lines into: timestamp, coordinates (DMS → decimal), agency
    for timestamp and coordinates:
        if only one of EXIF / stamp has it  → use that one
        if both agree                       → use EXIF, raise confidence
        if both disagree                    → use EXIF, lower confidence, record both
        if neither                          → value = null
save to the cache after every photo
```

**3. `extract_from_occurrence_summary.py`**
```
split the file into its fixed sections (header, GPS points, note, photos, audios, documents)
header lines "Key: value"        → facts (title, category, opened_at, ...)
GPS point lines                  → perimeter points
photo / audio lines              → timestamp + coordinates per file
document lines                   → document name + number
keep the free-text note as-is (step 4 handles it)
```

**4. `extract_from_field_notes.py`**
```
sources = each timestamped block of field-notes.md + each transcript segment from step 1
for each question group (people, areas, location, documents, equipment):
    ask qwen3-vl:4b: "extract only these fields, quote the passage for each,
                      leave out anything the sources don't state"
    (JSON schema enforced, temperature 0)
    for each value the model returns:
        search ALL sources for passages that actually contain that value
        if none found → drop the value (model invented or misread it)
        else          → keep it, citing every passage that contains it
        confidence = 0.82 if found in 2+ files, else 0.68
```

**5. `build_evidence.py`**
```
claims = everything from the caches of steps 2, 3 and 4
group claims by key (e.g. "car_registry", "owner_name")
for each key:
    if it's a list (team members, equipment) → one fact per distinct value
    if all readings agree                    → status "observed"
    if readings differ                       → status "conflicting", keep every reading
    confidence = best source's base score
               + bonus for each extra independent source
               − penalty if it cites a flagged whisper segment
add computed facts, e.g. reserve area = total suppressed − APP area   (status "inferred")
add every field the report needs that no source had                    (status "missing")
give each fact an id, write output/evidence.json
```

**6. `generate_report.py`**
```
load output/evidence.json   (never report_reference.json)
for each section of the printed form:
    for each fact key the section needs:
        observed / inferred → fill the sentence template with the value
        conflicting         → "sources disagree: reading A (conf), reading B (conf)"
        missing             → "the evidence does not support this field"
    render each sentence in Portuguese and English from the same fact
    section confidence = its lowest clause confidence; list its source files
self-check: every stated value must appear in its own sentence and cite a real fact
write output/report.json + output/report.txt
```

**7. `extract_from_final_report.py`**
```
write report_template.json (the form's empty sections)
for each non-boilerplate section of the form:
    ask qwen3-vl:4b with final_report.jpg: "copy this section verbatim, or return empty"
    if empty → leave the section out (means "unreadable", not "blank")
write report_reference.json (used only for scoring)
```

**8. `compare_report.py`**
```
for each section of the generated report:
    original section not readable        → NOT_COMPARABLE
    both blank                           → MATCH
    only the original has text           → MISSING_FROM_GENERATED
    only the generated report has text   → NOT_COMPARABLE
    dates disagree                       → CONFLICT
    all generated values found in the original  → MATCH
    some found                           → PARTIAL_MATCH
    none found                           → EXTRA_IN_GENERATED
also scan the original for hectares, document numbers, CPFs and names,
    and check whether the evidence supports each one
turn every self-check failure from step 6 into an UNSUPPORTED item
write output/comparison.json with counts and a completeness ratio
```

## Cost

On the Altamira case (10 voice notes, 4 photos), M1 Pro, CPU only, one run
takes about 2 minutes and costs R$0.

All case data is synthetic. Event rules: don't redistribute the data, delete it after the event, and don't publish coordinates.
