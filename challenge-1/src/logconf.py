"""
Shared logging setup for every script in src/.

Everything lands in one file, .logs/pipeline.log, across every script and
every run, rotated at 5 MB x 3 backups so a laptop nobody watches for weeks
doesn't quietly fill its disk. One shared file rather than one per script is
the point: run as a pipeline, the useful question afterwards is almost
always "what did this machine do this morning", not "what did one script
do in isolation" - a single file keeps that timeline in order.

Everything INFO and above goes to the file, but only WARNING and above also
reaches the console, so a real failure surfaces for someone who isn't tailing
the log without routine progress lines burying it.

Two env overrides, both for one run only:
    LOG_LEVEL=DEBUG           what is recorded at all
    CONSOLE_LOG_LEVEL=INFO    what also reaches stderr

CONSOLE_LOG_LEVEL exists because a caller that runs these scripts as
subprocesses - streamlit_app.py's pipeline button - has no sensible way to
read a shared rotating log file that other processes are writing to at the
same time. Lowering it makes each script narrate itself on stderr, which the
caller can attribute to the stage it started. The default is unchanged.
"""

import logging
import logging.handlers
import os
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / ".logs"
LOG_FILE = LOGS_DIR / "pipeline.log"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup(name):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # setup() called twice for this name - reuse, don't double-log

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT, _DATEFMT)

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(os.environ.get("CONSOLE_LOG_LEVEL", "WARNING").upper())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    return logger
