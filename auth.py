"""
auth.py - User authentication and management for Subtitrari
Stocheaza utilizatori in users.json cu parole hash-uite (SHA-256 + salt)
"""
import json
import os
import hashlib
import secrets
from pathlib import Path

USERS_FILE = Path(__file__).parent / "data" / "users.json"
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASS = "admin123"


def _hash_password(password: str, salt: str = None) -> tuple:
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return h, salt


def _load_users() -> dict:
    if USERS_FILE.exists():
        with open(USERS_FILE) as f:
            return json.load(f)
    # Create default admin on first run
    users = _create_default_admin()
    return users


def _save_users(users: dict):
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _create_default_admin() -> dict:
    h, salt = _hash_password(DEFAULT_ADMIN_PASS)
    users = {
        DEFAULT_ADMIN_USER: {
            "password_hash": h,
            "salt": salt,
            "role": "admin",
            "created_at": __import__("time").time(),
        }
    }
    _save_users(users)
    return users


def authenticate(username: str, password: str) -> bool:
    users = _load_users()
    user = users.get(username)
    if not user:
        return False
    h, _ = _hash_password(password, user["salt"])
    return h == user["password_hash"]


def is_admin(username: str) -> bool:
    users = _load_users()
    user = users.get(username)
    return user is not None and user.get("role") == "admin"


def add_user(username: str, password: str, role: str = "user") -> bool:
    if not username or not password:
        return False
    users = _load_users()
    if username in users:
        return False
    if not role:
        role = "user"
    h, salt = _hash_password(password)
    users[username] = {
        "password_hash": h,
        "salt": salt,
        "role": role,
        "created_at": __import__("time").time(),
    }
    _save_users(users)
    return True


def delete_user(username: str, caller: str) -> bool:
    if username == caller:
        return False
    users = _load_users()
    if username not in users:
        return False
    del users[username]
    _save_users(users)
    return True


def list_users(caller: str) -> list:
    users = _load_users()
    return [
        {"username": u, "role": d["role"], "created_at": d.get("created_at", 0)}
        for u, d in users.items()
    ]


def change_password(username: str, new_password: str, caller: str) -> bool:
    if username != caller:
        users = _load_users()
        caller_data = users.get(caller, {})
        if caller_data.get("role") != "admin":
            return False
    users = _load_users()
    if username not in users:
        return False
    h, salt = _hash_password(new_password)
    users[username]["password_hash"] = h
    users[username]["salt"] = salt
    _save_users(users)
    return True
