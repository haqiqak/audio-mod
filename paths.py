"""
paths.py — keep all runtime caches/data inside the project, and keep startup
memory low.

Import this FIRST — before nltk / sentence_transformers / torch / wordfreq —
in every entry module.  It points NLTK data and the HuggingFace/Torch model
caches at a project-local `.cache/` folder, so first-run downloads (WordNet,
tagger, cmudict, the SBERT model, torch hub) live next to the app instead of
scattered in per-user system cache directories.

setdefault() is used everywhere, so an explicit environment override (e.g. set
by a launcher or CI) always wins.
"""
import os
from pathlib import Path
import sys

_ROOT  = Path(__file__).resolve().parent          # project root
_CACHE = _ROOT / ".cache"
_VENDOR = _ROOT / ".vendor" / f"python{sys.version_info.major}{sys.version_info.minor}"

if _VENDOR.exists():
    sys.path.insert(0, str(_VENDOR))

# Cap BLAS/OpenMP threads BEFORE numpy/torch import.  OpenBLAS otherwise
# pre-allocates per-thread buffers sized for every CPU core, a large upfront
# allocation that fails on RAM-constrained machines ("OpenBLAS error: Memory
# allocation still failed").
#
# IMPORTANT: this was previously hardcoded to "1" thread everywhere. That
# avoided the load-time OOM, but it also single-threaded every CrisperWhisper
# generate() call — every encoder/decoder matmul forced onto one core — which
# is why a few seconds of audio could take 20+ minutes to transcribe on CPU.
# The OOM only happens once, at model load (OpenBLAS sizing per-core scratch
# buffers); it has nothing to do with how many threads are used during
# inference afterwards. So: cap at a small-but-real thread count instead of
# 1, which keeps load-time allocation bounded while letting inference
# actually use the CPU. setdefault still lets an explicit env var (e.g. a
# genuinely RAM-starved machine forcing "1") override this.
_DEFAULT_THREADS = str(max(1, min(4, (os.cpu_count() or 1))))
for _t in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_t, _DEFAULT_THREADS)

_TARGETS = {
    "NLTK_DATA":                  _CACHE / "nltk_data",
    "HF_HOME":                    _CACHE / "hf",
    "SENTENCE_TRANSFORMERS_HOME": _CACHE / "hf" / "sbert",
    "TORCH_HOME":                 _CACHE / "torch",
    # language_tool_python otherwise downloads its ~200 MB JAR to
    # ~/.cache/language_tool_python on C:. Keep it on the project drive too.
    "LTP_PATH":                   _CACHE / "language_tool",
}

for _var, _path in _TARGETS.items():
    _path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(_var, str(_path))

# Quiet, offline-friendly HF behaviour
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# hf-xet's transfer backend can silently hang on large model downloads on some
# networks (seen with the rephrase model on a fresh clone).  Fall back to plain
# HTTP so first-run downloads complete reliably.  Override by setting the var.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Ensure nltk searches the project data dir first (covers data already there).
try:
    import nltk
    _nltk_dir = os.environ["NLTK_DATA"]
    if _nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, _nltk_dir)
except Exception:
    pass
