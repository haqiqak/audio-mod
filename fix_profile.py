"""
fix_profile.py — one-time cleanup of the corrupted fluency profile.

Run from audio-mod root:
    python fix_profile.py

What it does:
  • Removes the phantom "I" onset (vowel-initial words were misbucketed
    there due to a bug in _onset_key — "I".isupper() was True, so the
    word "I" was stored as ARPAbet phoneme /I/ instead of empty onset).
  • Also removes any other single-vowel-letter onsets (A, E, O, U)
    that may have been misbucketed the same way.
  • Removes those onsets from onset_observations too so the EWMA
    starts fresh with correct data.
  • Cleans up session events that referenced those wrong onsets.
  • Re-saves the profile.
"""
import json
from pathlib import Path

USERS_DIR = Path("users")
PHANTOM_ONSETS = {"I", "A", "E", "O", "U"}   # single-vowel-letter phantom codes


def fix(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    username = data.get("username", path.stem)

    removed_risk   = {k for k in data.get("onset_risk", {})   if k in PHANTOM_ONSETS}
    removed_obs    = {k for k in data.get("onset_observations", {}) if k in PHANTOM_ONSETS}

    for k in removed_risk:
        del data["onset_risk"][k]
    for k in removed_obs:
        del data["onset_observations"][k]

    # Clean session events that referenced phantom onsets
    cleaned_sessions = 0
    for session in data.get("sessions", []):
        before = len(session.get("events", []))
        session["events"] = [
            e for e in session.get("events", [])
            if e.get("onset", "") not in PHANTOM_ONSETS
        ]
        session["count"] = len(session["events"])
        if len(session["events"]) < before:
            cleaned_sessions += 1

    # Recount total events
    data["event_count"] = sum(
        len(s.get("events", [])) for s in data.get("sessions", [])
    )

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{username}: removed onsets {removed_risk | removed_obs}, "
          f"cleaned {cleaned_sessions} session(s), "
          f"event_count now {data['event_count']}")


for p in USERS_DIR.glob("*.fluency_profile.json"):
    fix(p)
print("Done.")
