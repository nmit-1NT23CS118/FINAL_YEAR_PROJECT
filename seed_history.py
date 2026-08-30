"""
seed_history.py — OPTIONAL demo/dev utility, not part of the core pipeline
=============================================================================
The LSTM forecast needs 7 real days of history before it activates, which is
impractical to wait for during development or a viva demo. This script
backfills 6 synthetic-but-plausible prior days (today-6 .. today-1) into
weather_history.db, so that today's real submission through the app becomes
day 7 and the forecast unlocks immediately.

This is a convenience for demos only. For a "real" deployment that should
only ever use genuine daily readings, don't run this (or delete
weather_history.db afterwards to reset).

Usage:
    python seed_history.py
"""

import random
from datetime import date, timedelta

import history_store
from env_risk import compute_current_dpis

history_store.init_db()

LOCATION = input("Location/field name to seed history for (e.g. 'Ramesh - North Plot'): ").strip()
if not LOCATION:
    print("A location is required. Exiting.")
    raise SystemExit(1)

print(f"Seeding 6 synthetic prior days for '{LOCATION}' into weather_history.db ...")
for offset in range(6, 0, -1):
    d = (date.today() - timedelta(days=offset)).isoformat()
    temperature = round(random.uniform(14, 26), 1)
    humidity = round(random.uniform(60, 95), 1)
    rainfall = round(random.choice([0, 0, 0, random.uniform(1, 20)]), 1)
    soil_moisture = round(random.uniform(25, 60), 1)

    env_result = compute_current_dpis(temperature, humidity, rainfall, soil_moisture)
    dpis = {k: v["dpi"] for k, v in env_result.items()}

    history_store.upsert_today(LOCATION, temperature, humidity, rainfall, soil_moisture, dpis, reading_date=d)
    print(f"  {d}: T={temperature}  RH={humidity}  R={rainfall}  SM={soil_moisture}")

print(f"\nSeeded. '{LOCATION}' now has {history_store.count_days(LOCATION)} day(s) of history.")
print(f"Submit today's real weather through the app, using the SAME location name ('{LOCATION}'), "
      f"to record day 7 and unlock the forecast.")