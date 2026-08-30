"""
auth_store.py — Lightweight farmer accounts
===============================================
A minimal username/password system so each farmer's weather history is tied
to an authenticated account instead of a free-typed location string that
anyone could impersonate. Intentionally simple (SQLite + salted password
hashes via werkzeug, Flask session cookies) — this is proportionate to a
college project's threat model, not meant to replace a production-grade
auth system (no email verification, password reset, rate limiting, etc.).

Each user has exactly one farm_name, which is what history_store.py uses
as the partitioning key — so a user's weather history is automatically and
exclusively theirs, with no way for the client to claim a different one.
"""

import sqlite3
import os
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            farm_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def create_user(username: str, password: str, farm_name: str) -> dict:
    """
    Raises ValueError with a user-facing message on invalid input or if the
    username is already taken. Returns {"username", "farm_name"} on success.
    """
    username = (username or "").strip()
    password = password or ""
    farm_name = (farm_name or "").strip()

    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    if not farm_name:
        raise ValueError("Farm/field name is required.")

    conn = _connect()
    existing = conn.execute(
        "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE", (username,)
    ).fetchone()
    if existing:
        conn.close()
        raise ValueError("That username is already taken.")

    conn.execute(
        "INSERT INTO users (username, password_hash, farm_name, created_at) VALUES (?, ?, ?, ?)",
        (username, generate_password_hash(password), farm_name, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return {"username": username, "farm_name": farm_name}


def verify_user(username: str, password: str) -> dict | None:
    """Returns {"username", "farm_name"} if credentials are valid, else None."""
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ? COLLATE NOCASE", ((username or "").strip(),)
    ).fetchone()
    conn.close()
    if row is None or not check_password_hash(row["password_hash"], password or ""):
        return None
    return {"username": row["username"], "farm_name": row["farm_name"]}


def get_user(username: str) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ? COLLATE NOCASE", ((username or "").strip(),)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {"username": row["username"], "farm_name": row["farm_name"]}
