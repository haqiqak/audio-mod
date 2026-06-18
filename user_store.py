"""
user_store.py — file-based user storage for the standalone profiling app.

Each user is stored as  users/<username>.json  with the structure:

{
  "username": "alice",
  "password_hash": "<sha256 hex>",
  "preferences": {}
}

A separate  users/<username>.fluency_profile.json  is managed by
profiling.profile.SpeakerDifficultyProfile.

Public API
──────────
  list_users()                      -> list[str]
  user_exists(username)             -> bool
  register_user(username, password) -> (ok: bool, msg: str)
  verify_user(username, password)   -> (ok: bool, msg: str)
"""

import hashlib
import json
import re
from pathlib import Path

_USERS_DIR = Path(__file__).resolve().parent / "users"
_USERS_DIR.mkdir(exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe(username: str) -> str:
    return re.sub(r"[^a-z0-9_\-]", "", (username or "").lower())


def _path(username: str) -> Path:
    return _USERS_DIR / f"{_safe(username)}.json"


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _check(password: str, stored: str) -> bool:
    return _hash(password) == stored


def _read(username: str) -> dict | None:
    p = _path(username)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write(record: dict) -> None:
    p = _path(record["username"])
    with open(p, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


# ── public API ────────────────────────────────────────────────────────────────

def list_users() -> list[str]:
    return sorted(p.stem for p in _USERS_DIR.glob("*.json") if p.is_file())


def user_exists(username: str) -> bool:
    return _path(username).exists()


def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    if not username:
        return False, "Username cannot be empty."
    if not re.match(r"^[a-z0-9_\-]{2,32}$", username):
        return False, "Username must be 2–32 characters: a-z, 0-9, _ or -."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    if user_exists(username):
        return False, f"Username '{username}' is already taken."
    _write({"username": username, "password_hash": _hash(password), "preferences": {}})
    return True, ""


def verify_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    record = _read(username)
    if record is None:
        return False, "User not found."
    if not _check(password, record.get("password_hash", "")):
        return False, "Incorrect password."
    return True, ""
