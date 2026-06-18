"""
app.py — Audio Mod (Audio Module for Speech AI)

A light-themed, organized testing UI for the disfluency-detection pipeline.
Two screens only, by design, while the detection logic is still being
iterated on:
  1. Analyse — record / upload / demo, run the pipeline, see results
  2. Profile — the signed-in user's accumulated onset-risk data

Input modes:
  Demo    — instant, no audio, no model download
  Record  — browser mic via streamlit-mic-recorder
  Upload  — WAV/MP3/FLAC/M4A or JSON/TXT fixture
"""

import paths  # noqa: F401 — must be first (re-routes caches)

import json
import os
import tempfile
import threading
import time
import wave
from pathlib import Path

import streamlit as st

from auth import require_auth
from profiling.asr import CrisperWhisperASR, resample_to_16k
from profiling.detect import detect_disfluencies
from profiling.profile import SpeakerDifficultyProfile, profile_path

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Audio Mod",
    page_icon="🎚️",
    layout="wide",
    initial_sidebar_state="expanded",
)

require_auth()
CURRENT_USER: str = st.session_state.current_user


# ══════════════════════════════════════════════════════════════════════════════
# CACHED MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════
# Loads the 3.2GB CrisperWhisper model once per server process and reuses it
# across reruns/clicks instead of reconstructing it every time.
@st.cache_resource(show_spinner=False)
def get_asr(device: str) -> CrisperWhisperASR:
    return CrisperWhisperASR(device=device)


def _torch_diagnostics() -> str:
    """One-line CPU/torch diagnostic — thread count + MKL-DNN availability."""
    try:
        import torch
        threads = torch.get_num_threads()
        mkldnn = torch.backends.mkldnn.is_available()
        return f"torch threads={threads} · mkldnn={mkldnn}"
    except Exception as exc:
        return f"torch diagnostics unavailable ({exc})"


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS  —  light theme
# ══════════════════════════════════════════════════════════════════════════════
#   Background   #f7f8fb   page canvas, soft cool white
#   Surface      #ffffff   cards
#   Border       #e2e6ee   hairline borders on cards/inputs
#   Ink          #1b2433   primary text — near-black, not pure black
#   Ink-soft     #5b6478   secondary text
#   Accent       #2f6fed   primary action blue
#   Accent-soft  #eaf1ff   accent backgrounds
#   Success      #1f8a4c
#   Warning      #b5750a
#   Danger       #c23b3b
#   Log panel    #f7f9fd   light mono panel for the run log (was dark, now
#                          matches the rest of the UI per testing feedback)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Stop the browser from auto-dark-styling native form controls (selects,
   checkboxes, file picker, dropdown popovers) based on OS/browser dark
   mode. Without this, native widgets can render as solid black boxes that
   only become readable on hover/focus, regardless of our own CSS below. */
html { color-scheme: light only; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f7f8fb;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* kill Streamlit's default top padding so the header sits naturally */
[data-testid="stAppViewContainer"] > .main .block-container {
    padding-top: 1.6rem;
    max-width: 1100px;
}

/* ── universal text legibility ─────────────────────────────────────────── */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] div { color: #1b2433; }
h1, h2, h3, h4, h5 { color: #11172a !important; font-weight: 700 !important; }
[data-testid="stCaptionContainer"] p { color: #5b6478 !important; }

/* ── top brand bar ──────────────────────────────────────────────────────── */
.am-brandbar {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 1.1rem; margin-bottom: 1.3rem;
    border-bottom: 1px solid #e2e6ee;
}
.am-brand { display: flex; align-items: center; gap: .65rem; }
.am-mark {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #2f6fed, #6aa6ff);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.am-mark svg { width: 20px; height: 20px; }
.am-name { font-size: 1.32rem; font-weight: 700; color: #11172a; line-height: 1.1; }
.am-name span { color: #2f6fed; }
.am-tagline { font-size: .76rem; color: #8088a0; margin-top: .05rem; }
.am-user-chip {
    background: #eaf1ff; color: #1b4fb0; font-size: .8rem; font-weight: 600;
    padding: .35rem .85rem; border-radius: 20px; border: 1px solid #d5e4ff;
}

/* ── nav (radio styled as segmented control) ───────────────────────────── */
div[data-testid="stRadio"] > label { display: none; }
div[role="radiogroup"] {
    display: flex; gap: .4rem; background: #eef0f5; padding: .3rem;
    border-radius: 12px; width: fit-content;
}
div[role="radiogroup"] label {
    background: transparent; border-radius: 9px; padding: .45rem 1.1rem !important;
    margin: 0 !important; transition: background .12s, color .12s;
}
div[role="radiogroup"] label:hover { background: #e2e6f0; }
div[role="radiogroup"] label[data-checked="true"] {
    background: #ffffff; box-shadow: 0 1px 4px rgba(20,30,60,.12);
}
div[role="radiogroup"] p { font-weight: 600 !important; font-size: .88rem !important; }

/* ── cards ──────────────────────────────────────────────────────────────── */
.am-card {
    background: #ffffff; border: 1px solid #e2e6ee; border-radius: 14px;
    padding: 1.25rem 1.4rem; margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(20,30,60,.04);
}
.am-card h4 { margin: 0 0 .6rem; font-size: .95rem; color: #11172a !important; }
.am-card p { margin: 0; color: #3a4254 !important; }
.am-card-soft {
    background: #f7f9fd; border: 1px dashed #d2dcef; border-radius: 14px;
    padding: 1.4rem; text-align: center; color: #6b7488 !important;
}

/* ── pills / badges ─────────────────────────────────────────────────────── */
.am-pill {
    display: inline-block; padding: .22rem .7rem; border-radius: 20px;
    font-size: .73rem; font-weight: 700; margin: .12rem .18rem;
    color: #fff !important;
}
.am-pill-repetition     { background: #c2453b; }
.am-pill-block          { background: #b5750a; }
.am-pill-filler         { background: #6c4fc4; }
.am-pill-prolongation   { background: #1d6fb8; }
.am-pill-stutter_marker { background: #1f8a4c; }

/* ── stat tiles ─────────────────────────────────────────────────────────── */
.am-stats { display: flex; gap: .75rem; flex-wrap: wrap; margin: .6rem 0 1rem; }
.am-stat {
    flex: 1; min-width: 110px; background: #f7f9fd; border: 1px solid #e7ebf3;
    border-radius: 12px; padding: .8rem 1rem; text-align: center;
}
.am-stat .v { font-size: 1.55rem; font-weight: 700; color: #11172a; line-height: 1.1; }
.am-stat .l { font-size: .71rem; color: #8088a0; margin-top: .15rem; font-weight: 600;
              text-transform: uppercase; letter-spacing: .03em; }

/* ── transcript panel ───────────────────────────────────────────────────── */
.am-transcript {
    font-size: 1.08rem; line-height: 2; background: #ffffff;
    border: 1px solid #e2e6ee; border-radius: 14px; padding: 1.3rem 1.5rem;
}
.am-word-ok { color: #1b2433; }
.am-word-flag {
    background: #ffe3c2; border-radius: 5px; padding: .05rem .3rem;
    font-weight: 700; color: #8a3d00 !important; cursor: help;
}

/* ── risk bars ──────────────────────────────────────────────────────────── */
.am-risk-row { display: flex; align-items: center; gap: .6rem; margin: .3rem 0; }
.am-risk-label { width: 54px; font-size: .85rem; font-weight: 700; color: #1b2433; text-align: right; }
.am-risk-track { flex: 1; background: #eef0f5; border-radius: 8px; height: 14px; overflow: hidden; }
.am-risk-fill { height: 14px; border-radius: 8px; }
.am-risk-value { width: 40px; font-size: .82rem; font-weight: 600; color: #5b6478; }

/* ── log / console panel (light, matches the rest of the UI) ───────────── */
div.am-log, div.am-log * {
    background: #f7f9fd; color: #2a5a3a !important; font-family: 'JetBrains Mono', monospace;
    font-size: .79rem;
    line-height: 1.65; white-space: pre-wrap;
}
div.am-log {
    border: 1px solid #e2e6ee; border-radius: 10px; padding: .9rem 1.15rem;
    max-height: 230px; overflow-y: auto;
}

/* ── buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 9px; font-weight: 600; border: 1px solid #e2e6ee;
}
.stButton > button[kind="primary"] {
    background: #2f6fed; border-color: #2f6fed;
}
.stButton > button[kind="primary"]:hover { background: #2660d4; }

/* ── inputs ─────────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] {
    border-radius: 9px !important;
}

/* ── sidebar (kept light, not dark) ─────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #ffffff !important; border-right: 1px solid #e2e6ee;
}
section[data-testid="stSidebar"] * { color: #1b2433 !important; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #11172a !important; }

/* ── alerts: force readable text regardless of streamlit theme ──────────── */
[data-testid="stAlert"] p { color: inherit !important; }

/* ── expander header text ──────────────────────────────────────────────── */
[data-testid="stExpander"] summary p { color: #1b2433 !important; font-weight: 600; }
[data-testid="stExpander"] details { background: #ffffff; border: 1px solid #e2e6ee; border-radius: 10px; }

/* ── select / multiselect: closed box + the popover option list ─────────
   The popover list renders in a portal outside the main app container, so
   it needs its own explicit light styling rather than relying on the
   scoped rules above. */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background: #ffffff !important; color: #1b2433 !important;
    border: 1px solid #d8dde8 !important;
}
div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
    background: #ffffff !important;
}
li[role="option"], div[data-baseweb="menu"] li {
    background: #ffffff !important; color: #1b2433 !important;
}
li[role="option"]:hover, li[role="option"][aria-selected="true"] {
    background: #eaf1ff !important; color: #11172a !important;
}

/* ── file uploader drop zone ─────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    background: #f7f9fd !important; border: 1px dashed #d2dcef !important;
}
[data-testid="stFileUploaderDropzone"] * { color: #5b6478 !important; }

/* ── checkboxes / dataframe / code blocks ───────────────────────────────── */
[data-testid="stCheckbox"] label span { color: #1b2433 !important; }
[data-testid="stDataFrame"] { background: #ffffff !important; }
pre, code { background: #f7f9fd !important; color: #1b2433 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BRAND HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="am-brandbar">
  <div class="am-brand">
    <div class="am-mark">
      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="3" y="10" width="2.5" height="4" rx="1" fill="white"/>
        <rect x="7.5" y="6" width="2.5" height="12" rx="1" fill="white"/>
        <rect x="12" y="3" width="2.5" height="18" rx="1" fill="white"/>
        <rect x="16.5" y="7" width="2.5" height="10" rx="1" fill="white"/>
        <rect x="21" y="9.5" width="1.8" height="5" rx="0.9" fill="white" opacity="0.7"/>
      </svg>
    </div>
    <div>
      <div class="am-name">Audio <span>Mod</span></div>
      <div class="am-tagline">Audio Module for Speech AI — disfluency detection &amp; profiling</div>
    </div>
  </div>
  <div class="am-user-chip">{CURRENT_USER}</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Profile seed")
    st.caption("Sounds you find hardest to start words with.")
    raw_sounds = st.text_input("Onset sounds (comma-separated)", placeholder="b, pr, t, str", key="sr")
    if st.button("Apply", use_container_width=True):
        sounds = [s.strip() for s in raw_sounds.split(",") if s.strip()]
        _p = SpeakerDifficultyProfile.load(CURRENT_USER)
        _p.onboarding(sounds)
        _p.save()
        st.success(f"Seeded {len(sounds)} sound(s).")
        st.rerun()

    st.divider()
    st.markdown("### Session")
    if st.button("Sign out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_user = ""
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION (two screens only — Analyse, Profile)
# ══════════════════════════════════════════════════════════════════════════════
screen = st.radio("Screen", ["Analyse", "Profile"], horizontal=True, label_visibility="collapsed")
st.write("")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO FIXTURE
# ══════════════════════════════════════════════════════════════════════════════
DEMO = {"tokens": [
    {"word": "I",         "start": 0.00, "end": 0.18},
    {"word": "I",         "start": 0.18, "end": 0.36, "is_stutter": True},
    {"word": "want",      "start": 0.36, "end": 0.62},
    {"word": "to",        "start": 1.28, "end": 1.45},
    {"word": "uh",        "start": 1.45, "end": 1.72, "is_filler": True},
    {"word": "buy",       "start": 1.72, "end": 2.10},
    {"word": "buy-",      "start": 2.10, "end": 2.31, "is_stutter": True},
    {"word": "something", "start": 2.31, "end": 3.60},
    {"word": "special",   "start": 3.65, "end": 4.45},
]}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — duration estimate + threaded progress ticker
# ══════════════════════════════════════════════════════════════════════════════
def _estimate_wav_seconds(wav_bytes: bytes) -> float | None:
    try:
        import io
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


def _estimate_path_seconds(path: str) -> float | None:
    try:
        if Path(path).suffix.lower() not in {".wav", ".wave"}:
            return None
        with wave.open(path) as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


def _run_with_progress(fn, log, label: str, est_seconds: float | None):
    """Run fn() in a background thread, emitting elapsed-time updates to log."""
    result_box: dict = {}
    error_box: dict = {}

    def worker():
        try:
            result_box["value"] = fn()
        except Exception as exc:  # noqa: BLE001
            error_box["error"] = exc

    eta_note = (
        f" — est. {int(est_seconds)}s of audio, this may take a while on CPU"
        if est_seconds else ""
    )
    log(f"{label}{eta_note}")

    t0 = time.time()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    last_tick = 0
    while thread.is_alive():
        time.sleep(0.5)
        elapsed = time.time() - t0
        if elapsed - last_tick >= 4:
            last_tick = elapsed
            log(f"… still running ({int(elapsed)}s elapsed)")
    thread.join()

    if "error" in error_box:
        raise error_box["error"]
    log(f"Model finished in {time.time() - t0:.1f}s")
    return result_box.get("value")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(
    *,
    wav_bytes: bytes | None = None,
    fixture_dict: dict | None = None,
    txt_bytes: bytes | None = None,
    audio_path: str | None = None,
    device: str,
    save: bool,
    label: str,
    source_name: str,
):
    """Exactly one of wav_bytes/fixture_dict/txt_bytes/audio_path is set."""
    log_lines: list[str] = []
    log_ph = st.empty()

    def log(msg: str):
        log_lines.append(f"› {msg}")
        log_ph.markdown('<div class="am-log">' + "\n".join(log_lines) + "</div>", unsafe_allow_html=True)

    tmp_path: str | None = None
    try:
        _t0 = time.perf_counter()
        asr = get_asr(device)
        _t_get_asr = time.perf_counter() - _t0
        log(f"Model handle acquired in {_t_get_asr:.2f}s "
            f"({'cache hit' if _t_get_asr < 1 else 'SLOW — investigate'})")
        log(_torch_diagnostics())

        if fixture_dict is not None:
            log(f"Source: {source_name} (fixture — ASR skipped)")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
                json.dump(fixture_dict, tf)
                tmp_path = tf.name
            tokens = asr.transcribe(tmp_path)

        elif txt_bytes is not None:
            log(f"Source: {source_name} (text fixture — ASR skipped)")
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
                tf.write(txt_bytes)
                tmp_path = tf.name
            tokens = asr.transcribe(tmp_path)

        elif wav_bytes is not None:
            log(f"Source: {source_name} ({len(wav_bytes):,} bytes raw)")
            log("Resampling to 16 kHz mono…")
            resampled = resample_to_16k(wav_bytes)
            est = _estimate_wav_seconds(resampled)
            log(f"Loading CrisperWhisper · device={device}")
            tokens = _run_with_progress(
                lambda: asr.transcribe_bytes(resampled), log, "Transcribing", est,
            )

        else:  # audio_path
            log(f"Source: {source_name}")
            suffix = Path(audio_path).suffix.lower()
            if suffix in {".json", ".txt", ".transcript"}:
                log("Fixture detected — ASR skipped")
                tokens = asr.transcribe(audio_path)
            else:
                est = _estimate_path_seconds(audio_path)
                log(f"Loading CrisperWhisper · device={device}")
                tokens = _run_with_progress(
                    lambda: asr.transcribe(audio_path), log, "Transcribing", est,
                )

        log(f"Transcription complete — {len(tokens)} token(s)")

        if asr.last_timing:
            lp = asr.last_timing.get("load_pipeline_seconds")
            inf = asr.last_timing.get("inference_seconds")
            if lp is not None and inf is not None:
                log(f"Breakdown — pipeline load: {lp:.2f}s · inference: {inf:.2f}s")

        log("Running disfluency detector…")
        events = detect_disfluencies(tokens)
        log(f"Detection complete — {len(events)} event(s)")

        if save:
            if events:
                p = SpeakerDifficultyProfile.load(CURRENT_USER)
                p.update(events, session_id=label.strip() or None)
                p.save()
                log(f"Profile updated for '{CURRENT_USER}'")
            else:
                log("No events — profile unchanged")
        else:
            log("Dry run — profile not saved")

        # ── results ──────────────────────────────────────────────────────────
        st.markdown("#### Transcript")
        dis_idx = {e["index"] for e in events}
        ev_map: dict[int, list] = {}
        for e in events:
            ev_map.setdefault(e["index"], []).append(e)

        parts = []
        for i, tok in enumerate(tokens):
            w = tok.get("word", "")
            if i in dis_idx:
                tip = " / ".join({e["type"] for e in ev_map[i]})
                parts.append(f'<span class="am-word-flag" title="{tip}">{w}</span>')
            else:
                parts.append(f'<span class="am-word-ok">{w}</span>')
        st.markdown('<div class="am-transcript">' + " ".join(parts) + "</div>", unsafe_allow_html=True)
        st.caption("Highlighted words were flagged as disfluencies — hover to see the type.")

        n_tok, n_ev = len(tokens), len(events)
        fluency = round(100 * (1 - n_ev / max(n_tok, 1)), 1)
        tc: dict[str, int] = {}
        for e in events:
            tc[e["type"]] = tc.get(e["type"], 0) + 1

        st.markdown(
            f'<div class="am-stats">'
            f'<div class="am-stat"><div class="v">{n_tok}</div><div class="l">Tokens</div></div>'
            f'<div class="am-stat"><div class="v">{n_ev}</div><div class="l">Disfluencies</div></div>'
            f'<div class="am-stat"><div class="v">{fluency}%</div><div class="l">Fluency rate</div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        if tc:
            badges = "".join(
                f'<span class="am-pill am-pill-{t}">{t.replace("_", " ")} × {c}</span>'
                for t, c in sorted(tc.items())
            )
            st.markdown(badges, unsafe_allow_html=True)

        with st.expander("Event table", expanded=bool(events)):
            if events:
                st.dataframe(
                    [{
                        "Word": e["word"],
                        "Type": e["type"],
                        "Confidence": f"{e['confidence']:.2f}",
                        "Start (s)": f"{e['start']:.2f}" if e.get("start") is not None else "—",
                        "End (s)": f"{e['end']:.2f}" if e.get("end") is not None else "—",
                        "Evidence": e.get("evidence", ""),
                    } for e in events],
                    width="stretch", hide_index=True,
                )
            else:
                st.success("No disfluencies detected.")

        with st.expander("Raw token JSON", expanded=False):
            st.json(tokens)

    except RuntimeError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        import traceback
        st.code(traceback.format_exc(), language="python")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: ANALYSE
# ══════════════════════════════════════════════════════════════════════════════
if screen == "Analyse":

    try:
        from streamlit_mic_recorder import mic_recorder as _mic_rec
        HAS_MIC = True
    except ImportError:
        HAS_MIC = False

    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.markdown("##### Input")
        mode = st.radio(
            "Input mode", ["Demo", "Record", "Upload"],
            horizontal=True, label_visibility="collapsed", key="input_mode",
        )

    with col_right:
        st.markdown("##### Options")
        device = st.selectbox("ASR device", ["cpu", "cuda", "mps"], label_visibility="collapsed")

    o1, o2 = st.columns([1, 2])
    with o1:
        save = st.checkbox("Save to profile", value=True)
    with o2:
        label = st.text_input("Session label", placeholder="optional label", label_visibility="collapsed")

    st.write("")

    if mode == "Demo":
        st.markdown(
            '<div class="am-card"><h4>Instant demo</h4>'
            "<p>Loads a preset 9-token sequence with no recording and no model "
            "download. Use this to confirm the pipeline end-to-end in under a second.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("Run demo", type="primary"):
            run_pipeline(fixture_dict=DEMO, device=device, save=save, label=label or "demo", source_name="demo fixture")

    elif mode == "Record":
        if HAS_MIC:
            st.caption("Click to start recording, click again to stop. Analysis runs automatically.")
            audio = _mic_rec(
                key="mic", start_prompt="Start recording", stop_prompt="Stop & analyse",
                use_container_width=True, format="wav",
            )
            if audio and audio.get("bytes"):
                st.audio(audio["bytes"], format="audio/wav")
                run_pipeline(
                    wav_bytes=audio["bytes"], device=device, save=save,
                    label=label or "mic-session",
                    source_name=f"microphone ({len(audio['bytes']):,} bytes)",
                )
        else:
            st.warning("streamlit-mic-recorder is not installed. Run `pip install streamlit-mic-recorder` and restart.")

    else:  # Upload
        uploaded = st.file_uploader(
            "Upload audio or a fixture",
            type=["wav", "mp3", "flac", "m4a", "ogg", "webm", "json", "txt", "transcript"],
            label_visibility="collapsed",
        )
        st.caption("JSON/TXT fixtures run instantly. Audio files use CrisperWhisper.")

        if uploaded:
            suffix = Path(uploaded.name).suffix.lower()
            if st.button("Analyse", type="primary"):
                raw = uploaded.read()
                if suffix == ".json":
                    run_pipeline(
                        fixture_dict=json.loads(raw.decode("utf-8")),
                        device=device, save=save, label=label or uploaded.name, source_name=uploaded.name,
                    )
                elif suffix in {".txt", ".transcript"}:
                    run_pipeline(
                        txt_bytes=raw, device=device, save=save,
                        label=label or uploaded.name, source_name=uploaded.name,
                    )
                else:
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                        tf.write(raw)
                        tmp = tf.name
                    try:
                        run_pipeline(
                            audio_path=tmp, device=device, save=save,
                            label=label or uploaded.name, source_name=uploaded.name,
                        )
                    finally:
                        try:
                            os.unlink(tmp)
                        except OSError:
                            pass


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: PROFILE
# ══════════════════════════════════════════════════════════════════════════════
else:
    p = SpeakerDifficultyProfile.load(CURRENT_USER)

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        sounds_str = ", ".join(p.self_reported_sounds) or "—"
        st.markdown(
            f'<div class="am-card"><h4>Overview</h4>'
            f'<div class="am-stats">'
            f'<div class="am-stat"><div class="v">{len(p.sessions)}</div><div class="l">Sessions</div></div>'
            f'<div class="am-stat"><div class="v">{p.event_count}</div><div class="l">Events</div></div>'
            f"</div>"
            f'<p style="margin-top:.5rem;font-size:.85rem;">Self-reported sounds: '
            f"<strong>{sounds_str}</strong></p></div>",
            unsafe_allow_html=True,
        )

    with col2:
        top = p.top_onsets(12)
        if top:
            st.markdown('<div class="am-card"><h4>Top onset risks</h4>', unsafe_allow_html=True)
            for onset, risk in top:
                pct = int(risk * 100)
                color = "#c2453b" if risk >= 0.55 else ("#b5750a" if risk >= 0.35 else "#1f8a4c")
                st.markdown(
                    f'<div class="am-risk-row">'
                    f'<span class="am-risk-label">{onset}</span>'
                    f'<div class="am-risk-track"><div class="am-risk-fill" '
                    f'style="width:{pct}%;background:{color};"></div></div>'
                    f'<span class="am-risk-value">{risk:.2f}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="am-card-soft">No data yet — run an analysis on the Analyse screen.</div>',
                unsafe_allow_html=True,
            )

    with st.expander("All onset values", expanded=False):
        if p.onset_risk:
            st.dataframe(
                [{
                    "Onset": k, "Risk": round(v, 4),
                    "Events seen": p.onset_observations.get(k, {}).get("events", 0),
                    "Disfluent": p.onset_observations.get(k, {}).get("disfluent", 0),
                } for k, v in sorted(p.onset_risk.items(), key=lambda x: -x[1])],
                width="stretch", hide_index=True,
            )

    with st.expander("Raw profile JSON", expanded=False):
        st.json(p.to_dict())

    st.write("")
    if st.button("Delete my profile data"):
        st.session_state["confirm_reset"] = True

    if st.session_state.get("confirm_reset"):
        st.warning("This deletes all session history and onset scores. This cannot be undone.")
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("Yes, delete everything", type="primary"):
                pp = profile_path(CURRENT_USER)
                if pp.exists():
                    pp.unlink()
                st.session_state["confirm_reset"] = False
                st.rerun()
        with cc2:
            if st.button("Cancel"):
                st.session_state["confirm_reset"] = False
                st.rerun()
