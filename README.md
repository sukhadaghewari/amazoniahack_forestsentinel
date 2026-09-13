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

## What each step does

1. **Transcribe voice notes.** Whisper transcribes each voice note in
   Portuguese and skips silence. Each segment gets a confidence score, and
   segments that look unreliable (silence, low confidence, repeated text) are
   flagged.
2. **Read photo stamps.** Reads the time and GPS from each photo's EXIF
   metadata, then OCRs the stamp burned into the bottom of the image. If the
   two readings agree, confidence goes up. If they disagree, both are kept and
   confidence goes down.
3. **Parse the app export.** A line parser reads the occurrence summary's
   fixed sections: header fields, GPS points, photo and audio positions, and
   issued documents. No model is used.
4. **Extract facts from text.** The field notes and the step 1 transcripts are
   sent to Qwen3-VL in five small questions: people, areas, location,
   documents, equipment. A value is kept only if it actually appears in one of
   the sources. A value found in two different files gets higher confidence.
5. **Build evidence.** Claims about the same thing from steps 2–4 are merged
   into one fact, marked `observed` (sources agree), `conflicting` (every
   reading kept), `inferred` (computed, e.g. area totals) or `missing` (no
   source). Confidence comes from how many independent sources back the fact,
   minus a penalty for citing a flagged audio segment.
6. **Draft the report.** Each section of the official form is filled from the
   facts with fixed sentence templates, in Portuguese and English. A missing
   fact is stated as unsupported, and a conflicting one shows every reading. A
   self-check confirms every stated value appears in its sentence and cites a
   real fact.
7. **Read the filed report.** Qwen3-VL copies each section of the photographed
   official report word for word. It leaves out any section it can't read. The
   result is used only for scoring, never for drafting.
8. **Score against it.** Each section of the generated report is compared
   with the filed one and labelled MATCH, PARTIAL_MATCH, CONFLICT,
   MISSING_FROM_GENERATED, EXTRA_IN_GENERATED, UNSUPPORTED or NOT_COMPARABLE.
   It also checks whether the evidence backs the areas, document numbers and
   names written in the original.

## Cost

On the Altamira case (10 voice notes, 4 photos), M1 Pro, CPU only, one run
takes about 2 minutes and costs R$0.

All case data is synthetic. Event rules: don't redistribute the data, delete it after the event, and don't publish coordinates.
