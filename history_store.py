"""
history_store.py — Rolling daily weather+DPI history for the LSTM forecaster
==============================================================================
The multi-disease LSTM (multi_disease_lstm_v2.keras) was trained on 7
consecutive days of (4 weather + 8 current-day DPI) readings to forecast 5
days ahead. The Flask form only collects *today's* single reading, so this
module persists each day's reading to a local SQLite DB (weather_history.db,
created next to this file) and exposes the last 7 real days once they've
accumulated — separately for each farmer/field.

Rows are keyed on (location, date) — one row per farm/field per calendar
day. Multiple farmers using the app on the same day get independent rows,
so one farmer's submission never overwrites another's. `location` is a
free-text identifier the farmer enters once and reuses (e.g. "Ramesh -
North Plot") — see app.py for how it's collected.

Resubmitting the form multiple times today for the same location just
overwrites that location's row for today (upsert). It does NOT fabricate
extra history; the forecast genuinely stays locked, per location, until 7
distinct days have been recorded for that specific location (see
seed_history.py for an optional demo shortcut).
"""

import sqlite3
import os
from datetime import date

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weather_history.db")

# Must match the DISEASES order in ev_project/models/train_lstm.py exactly —
# this is also the key order env_risk.py's DISEASE_MODELS dict yields.
DISEASES = [
    "tomato_late_blight", "grape_downy_mildew", "wheat_leaf_rust", "apple_scab",
    "tomato_early_blight", "potato_early_blight", "pepper_bacterial_spot", "potato_late_blight",
]


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    dpi_cols = ", ".join(f"DPI_{d} REAL" for d in DISEASES)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS readings (
            location TEXT NOT NULL,
            date TEXT NOT NULL,
            temperature REAL,
            humidity REAL,
            rainfall REAL,
            soil_moisture REAL,
            {dpi_cols},
            PRIMARY KEY (location, date)
        )
    """)
    conn.commit()
    conn.close()
    _migrate_legacy_table(conn=_connect())


def _migrate_legacy_table(conn):
    """
    One-time safety net: if an old pre-multi-tenant weather_history.db exists
    (schema: date TEXT PRIMARY KEY, no location column), its rows are not
    attributable to any specific farmer and must not be silently merged into
    the new table. We rename the old table instead of deleting it, so no
    data is lost, and log what happened.
    """
    try:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(readings)")]
        if "location" not in cols:
            # Old single-tenant table shape shouldn't exist alongside the
            # new CREATE TABLE IF NOT EXISTS above, but guard anyway.
            conn.execute("ALTER TABLE readings RENAME TO readings_legacy_no_location")
            conn.commit()
            print("[history_store] Found an old readings table with no 'location' column. "
                  "Renamed it to 'readings_legacy_no_location' — its data is NOT per-farmer "
                  "and was not migrated automatically. A fresh 'readings' table was created.")
    except sqlite3.OperationalError:
        pass  # table didn't exist yet, nothing to migrate
    finally:
        conn.close()


def upsert_today(location: str, temperature: float, humidity: float, rainfall: float,
                  soil_moisture: float, dpis: dict, reading_date: str | None = None):
    """
    location: farmer/field identifier — required, keeps this farmer's history separate.
    dpis: {disease_key: dpi_value} for all keys in DISEASES.
    reading_date: ISO date string; defaults to the server's today.
    """
    if not location or not location.strip():
        raise ValueError("location is required to store a reading")
    location = location.strip()

    d = reading_date or date.today().isoformat()
    conn = _connect()
    cols = ["location", "date", "temperature", "humidity", "rainfall", "soil_moisture"] + \
           [f"DPI_{k}" for k in DISEASES]
    vals = [location, d, temperature, humidity, rainfall, soil_moisture] + [dpis[k] for k in DISEASES]
    placeholders = ", ".join("?" for _ in cols)
    col_list = ", ".join(cols)
    conn.execute(f"INSERT OR REPLACE INTO readings ({col_list}) VALUES ({placeholders})", vals)
    conn.commit()
    conn.close()


def get_history(location: str, n: int = 7):
    """Returns up to the most recent n rows for this location, ordered oldest -> newest."""
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM readings WHERE location = ? ORDER BY date DESC LIMIT ?",
        (location, n),
    ).fetchall()
    conn.close()
    return list(reversed(rows))


def count_days(location: str) -> int:
    conn = _connect()
    n = conn.execute(
        "SELECT COUNT(*) as c FROM readings WHERE location = ?", (location,)
    ).fetchone()["c"]
    conn.close()
    return n


def list_locations():
    """Returns all distinct location identifiers currently stored. Handy for debugging/demo."""
    conn = _connect()
    rows = conn.execute("SELECT DISTINCT location FROM readings ORDER BY location").fetchall()
    conn.close()
    return [r["location"] for r in rows]