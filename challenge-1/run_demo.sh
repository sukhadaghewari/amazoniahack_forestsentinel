#!/usr/bin/env bash
# The whole demo, one command:
#   Streamlit UI      http://localhost:8501
#   API + Swagger     http://127.0.0.1:8000/docs
#
# No model is loaded by either process. The extractors have already written
# data/<case>/.cache/, build_evidence.py consolidated it into
# data/<case>/output/evidence.json, and drafting the report from those facts is
# deterministic - no model call, so nothing here is slow or nondeterministic.
#
# The UI calls generate_report in-process and does not use the API, so the demo
# still works if the API fails to start; the API is there for integration and
# for the interactive docs.
set -euo pipefail

cd "$(dirname "$0")"

API_PORT="${API_PORT:-8000}"
PYTHON="../.amazonia/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

cleanup() {
    kill "${API_PID:-}" "${UI_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

(cd src && "../$PYTHON" -m uvicorn api:app --host 127.0.0.1 --port "$API_PORT") &
API_PID=$!

"$PYTHON" -m streamlit run streamlit_app.py &
UI_PID=$!

wait -n "$API_PID" "$UI_PID"
