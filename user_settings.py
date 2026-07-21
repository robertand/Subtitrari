"""
user_settings.py - Gestionează setările persistente per utilizator pe server.
"""
import json
import time
from pathlib import Path

SETTINGS_FILE = Path(__file__).parent / "data" / "user_settings.json"


def _load_all() -> dict:
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return {}


def _save_all(data: dict):
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_settings(username: str) -> dict:
    all_s = _load_all()
    return all_s.get(username, {})


def save_settings(username: str, settings: dict):
    all_s = _load_all()
    all_s[username] = settings
    all_s[username]["_updated_at"] = time.time()
    _save_all(all_s)


GLOBAL_KEY = "__global__"


def get_global_setting(key: str, default=None):
    all_s = _load_all()
    return all_s.get(GLOBAL_KEY, {}).get(key, default)


def set_global_setting(key: str, value):
    all_s = _load_all()
    if GLOBAL_KEY not in all_s:
        all_s[GLOBAL_KEY] = {}
    all_s[GLOBAL_KEY][key] = value
    all_s[GLOBAL_KEY]["_updated_at"] = time.time()
    _save_all(all_s)
