"""
lstm_forecast.py — 5-day-ahead multi-disease DPI forecasting
==============================================================
Loads the trained multi-output LSTM (multi_disease_lstm_v2.keras) and its
paired MinMaxScalers (multi_scalers_v2.pkl) once, and exposes forecast_dpis()
to turn 7 days of history rows into an 8-disease, 5-day-ahead DPI forecast.

Both files must be copied into this folder (next to app.py) from
ev_project/models/ after running ev_project/models/train_lstm.py:
  - multi_disease_lstm_v2.keras
  - multi_scalers_v2.pkl

If they're missing, this module degrades gracefully: is_available() returns
False and forecast_dpis() returns None, so app.py can still serve image +
today's-DPI diagnoses without the forecast feature.
"""

import os
import pickle
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi_disease_lstm_v2.keras")
SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi_scalers_v2.pkl")

_model = None
_scaler_X = None
_scaler_y = None
_feature_cols = None
_target_cols = None
_available = False


def _lazy_load():
    global _model, _scaler_X, _scaler_y, _feature_cols, _target_cols, _available

    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        print(f"[lstm_forecast] Model/scaler not found next to app.py — forecast disabled.\n"
              f"  Expected: {MODEL_PATH}\n"
              f"  and:      {SCALER_PATH}\n"
              f"  Run ev_project/models/train_lstm.py, then copy both output files here.")
        return

    try:
        import tensorflow as tf  # imported lazily so the app still starts without TF installed
        with open(SCALER_PATH, "rb") as f:
            bundle = pickle.load(f)
        _scaler_X = bundle["scaler_X"]
        _scaler_y = bundle["scaler_y"]
        _feature_cols = bundle["feature_cols"]
        _target_cols = bundle["target_cols"]
        _model = tf.keras.models.load_model(MODEL_PATH)
        _available = True
        print("[lstm_forecast] Multi-disease LSTM loaded — 5-day forecasting enabled.")
    except Exception as e:
        print(f"[lstm_forecast] Failed to load LSTM/scalers — forecast disabled. Error: {e}")


_lazy_load()


def is_available() -> bool:
    return _available


def forecast_dpis(history_rows):
    """
    history_rows: list of >=7 sqlite3.Row (or dict-like) objects, oldest -> newest,
    each with temperature, humidity, rainfall, soil_moisture, and DPI_<disease>
    for all 8 diseases (see history_store.DISEASES for the key order).

    Returns {disease_key: forecast_dpi_float} (5 days ahead) or None if the
    model isn't loaded or fewer than 7 rows were supplied.
    """
    if not _available or len(history_rows) < 7:
        return None

    last7 = history_rows[-7:]
    seq = np.array([[row[col] for col in _feature_cols] for row in last7], dtype=float)
    seq_scaled = _scaler_X.transform(seq)
    seq_scaled = seq_scaled.reshape(1, 7, len(_feature_cols))

    pred_scaled = _model.predict(seq_scaled, verbose=0)
    pred = _scaler_y.inverse_transform(pred_scaled)[0]

    return {col.replace("DPI_", ""): float(round(val, 2)) for col, val in zip(_target_cols, pred)}
