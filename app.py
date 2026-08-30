"""
app.py  -  Plant Disease Detection Web Interface
=================================================
Run:  python app.py
Then open:  http://localhost:5000

Step 3 addition: alongside the existing image + today's-weather combined
diagnosis (FR4), this now also attaches a 5-day-ahead LSTM forecast for the
same disease, once 7 real days of weather have been recorded via
history_store.py. See lstm_forecast.py and history_store.py for details, and
seed_history.py for an optional demo shortcut.
"""

import os, io, base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, send_from_directory, session

from env_risk import compute_current_dpis, get_risk_label
import history_store
import lstm_forecast
import auth_store

app = Flask(__name__, static_folder="static")

# Sessions need a secret key to sign cookies. Setting FLASK_SECRET_KEY as a
# real environment variable is recommended so logins survive server
# restarts; without it, a random key is generated each startup and every
# farmer is logged out whenever the server restarts (fine for a demo, not
# for real use).
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "plant_disease_efficientnet_b0.pth"
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy",
]
TREATMENT = {
    "Pepper__bell___Bacterial_spot": "Bacterial spot detected. Use disease-free seeds, apply copper-based bactericide (e.g. Kocide 3000) every 7-10 days, and remove infected leaves. Avoid overhead irrigation.",
    "Pepper__bell___healthy":        "Plant is healthy! Maintain balanced fertilisation (N-P-K), water at the base, and monitor for pests regularly.",
    "Potato___Early_blight":         "Early blight (Alternaria) detected. Remove infected lower leaves, apply fungicide containing chlorothalonil or mancozeb, and ensure adequate plant spacing for airflow.",
    "Potato___Late_blight":          "Late blight (Phytophthora) – act urgently! Apply systemic fungicide (metalaxyl + mancozeb) immediately. Destroy severely infected plants. Avoid wet foliage and improve drainage.",
    "Potato___healthy":              "Plant is healthy! Keep soil consistently moist, hill up soil around stems, and watch for early blight symptoms as plants mature.",
    "Tomato_Early_blight":           "Early blight detected. Remove affected leaves, apply copper-based or chlorothalonil fungicide, stake plants for better air circulation, and mulch around the base.",
    "Tomato_Late_blight":            "Late blight detected – high risk of total crop loss! Apply fungicide (cymoxanil/mancozeb) and repeat every 5-7 days. Remove heavily infected plants.",
    "Tomato_healthy":                "Plant is healthy! Ensure consistent watering, support with stakes, and feed with potassium-rich fertiliser during fruiting.",
}
SEVERITY = {
    "Pepper__bell___Bacterial_spot": "warning",
    "Pepper__bell___healthy":        "healthy",
    "Potato___Early_blight":         "warning",
    "Potato___Late_blight":          "danger",
    "Potato___healthy":              "healthy",
    "Tomato_Early_blight":           "warning",
    "Tomato_Late_blight":            "danger",
    "Tomato_healthy":                "healthy",
}

# Groups classes by plant species. Used only when the user optionally tells us
# which plant it is, so we can restrict predictions to that plant's classes
# and eliminate cross-species confusion (e.g. Potato vs Tomato Early Blight).
PLANT_CLASSES = {
    "Pepper": ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy"],
    "Potato": ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"],
    "Tomato": ["Tomato_Early_blight", "Tomato_Late_blight", "Tomato_healthy"],
}

# FR4 — maps each image class to the environmental disease key it should be
# combined with. env_risk.py covers all 8 classes' actual diseases.
DISEASE_MAP = {
    "Tomato_Late_blight":            "tomato_late_blight",
    "Potato___Late_blight":          "potato_late_blight",
    "Tomato_Early_blight":           "tomato_early_blight",
    "Potato___Early_blight":         "potato_early_blight",
    "Pepper__bell___Bacterial_spot": "pepper_bacterial_spot",
    "Tomato_healthy":                "tomato_late_blight",     # preventive check
    "Potato___healthy":              "potato_late_blight",     # preventive check
    "Pepper__bell___healthy":        "pepper_bacterial_spot",  # preventive check
}

DISEASE_DISPLAY_NAMES = {
    "tomato_late_blight":    "Late Blight (Phytophthora infestans)",
    "potato_late_blight":    "Late Blight (Phytophthora infestans)",
    "tomato_early_blight":   "Early Blight (Alternaria solani)",
    "potato_early_blight":   "Early Blight (Alternaria solani)",
    "pepper_bacterial_spot": "Bacterial Spot (Xanthomonas spp.)",
}


def combine_diagnosis(predicted_class: str, image_severity: str, env_key: str, env_result: dict) -> dict:
    """
    FR4 — combines the image-based prediction with the environmental risk
    for whichever disease applies, into one final diagnosis.
    """
    if env_key is None:
        return {
            "available": False,
            "note": "Environmental risk model is not yet available for this disease.",
        }

    dpi  = env_result[env_key]["dpi"]
    risk = env_result[env_key]["risk"]  # "Low" / "Medium" / "High"
    is_diseased = image_severity in ("warning", "danger")

    if is_diseased and risk == "High":
        combined_severity = "danger"
        message = ("Disease detected AND current environmental conditions strongly favor "
                   "continued spread. Immediate action is strongly recommended.")
    elif is_diseased and risk == "Medium":
        combined_severity = "danger" if image_severity == "danger" else "warning"
        message = ("Disease detected. Environmental conditions are moderately favorable for "
                   "further spread — treat promptly and continue monitoring.")
    elif is_diseased and risk == "Low":
        combined_severity = image_severity
        message = ("Disease detected, but current environmental conditions are not favorable "
                   "for rapid spread. Standard treatment is still recommended.")
    elif (not is_diseased) and risk == "High":
        combined_severity = "warning"
        message = ("No visible symptoms yet, but current environmental conditions strongly favor "
                   "disease development. Preventive early warning — inspect closely over the next "
                   "few days and consider a protective spray.")
    elif (not is_diseased) and risk == "Medium":
        combined_severity = "healthy"
        message = "No symptoms detected. Environmental risk is moderate — continue routine monitoring."
    else:
        combined_severity = "healthy"
        message = "No symptoms detected and conditions are unfavorable for disease development."

    return {
        "available":          True,
        "disease_key":        env_key,
        "disease_name":       DISEASE_DISPLAY_NAMES.get(env_key, env_key),
        "dpi":                dpi,
        "environmental_risk": risk,
        "combined_severity":  combined_severity,
        "message":            message,
    }


def build_forecast_block(location: str, env_key: str, current_dpi: float) -> dict:
    """
    Step 3 addition — 5-day-ahead forecast for the same disease as the
    combined diagnosis above, using this location's stored weather history.
    Always returns a dict describing forecast status, even when not ready,
    so the frontend can show a "collecting history X/7" state instead of nothing.
    """
    days_recorded = history_store.count_days(location)

    if not lstm_forecast.is_available():
        return {
            "status": "unavailable",
            "message": ("5-day forecasting isn't set up yet on this server — copy "
                        "multi_disease_lstm_v2.keras and multi_scalers_v2.pkl into the "
                        "project folder to enable it."),
        }

    if days_recorded < 7:
        return {
            "status": "collecting",
            "days_recorded": days_recorded,
            "days_needed": 7,
            "message": (f"Collecting weather history for the 5-day forecast for this field — "
                        f"{days_recorded}/7 days recorded. Submit today's reading daily, using "
                        f"the same location name, to unlock it."),
        }

    history_rows = history_store.get_history(location, 7)
    forecast_all = lstm_forecast.forecast_dpis(history_rows)

    if forecast_all is None or env_key not in forecast_all:
        return {
            "status": "collecting",
            "days_recorded": days_recorded,
            "days_needed": 7,
            "message": "Forecast could not be generated for this disease yet.",
        }

    forecast_dpi = forecast_all[env_key]
    forecast_risk = get_risk_label(forecast_dpi)

    delta = forecast_dpi - current_dpi
    if delta > 5:
        trend = "rising"
    elif delta < -5:
        trend = "falling"
    else:
        trend = "steady"

    trend_phrase = {
        "rising":  "rising",
        "falling": "easing",
        "steady":  "holding steady",
    }[trend]

    return {
        "status":        "ready",
        "disease_key":   env_key,
        "disease_name":  DISEASE_DISPLAY_NAMES.get(env_key, env_key),
        "current_dpi":   current_dpi,
        "forecast_dpi":  forecast_dpi,
        "forecast_risk": forecast_risk,
        "trend":         trend,
        "message": (f"Based on the last 7 days of recorded weather, risk is projected to be "
                    f"{trend_phrase} — DPI ~{forecast_dpi} ({forecast_risk} risk) in 5 days, "
                    f"versus {current_dpi} today."),
    }


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, len(CLASS_NAMES)),
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print(f"Model loaded on {device} ")

# ── INIT HISTORY DB (safe to call every startup — CREATE TABLE IF NOT EXISTS) ──
history_store.init_db()
auth_store.init_db()

# ── AUTH ROUTES ──────────────────────────────────────────────────────────────
@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    try:
        user = auth_store.create_user(
            data.get("username", ""), data.get("password", ""), data.get("farm_name", "")
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    session["username"] = user["username"]
    return jsonify({"logged_in": True, **user})


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    user = auth_store.verify_user(data.get("username", ""), data.get("password", ""))
    if user is None:
        return jsonify({"error": "Incorrect username or password."}), 401
    session["username"] = user["username"]
    return jsonify({"logged_in": True, **user})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.pop("username", None)
    return jsonify({"logged_in": False})


@app.route("/api/me")
def me():
    username = session.get("username")
    if not username:
        return jsonify({"logged_in": False})
    user = auth_store.get_user(username)
    if user is None:
        # Account was deleted server-side but an old session cookie remains.
        session.pop("username", None)
        return jsonify({"logged_in": False})
    return jsonify({"logged_in": True, **user})


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/plants")
def plants():
    """Returns the plant options for the optional dropdown in the UI."""
    return jsonify({"plants": list(PLANT_CLASSES.keys())})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Optional hint from the user: "Pepper", "Potato", or "Tomato".
    plant_type = request.form.get("plant_type", "").strip()
    plant_filter_applied = None

    # FR1 — today's environmental reading, all optional. If all four are
    # provided, we compute the FR4 combined diagnosis + attempt a forecast.
    # The location/field identifier is NEVER taken from client-submitted
    # form data — it's derived from the logged-in user's account, so one
    # farmer can never write into another farmer's history by typing their
    # name (see auth_store.py). Weather submission therefore requires login;
    # image-only diagnosis does not.
    env_fields = ["temperature", "humidity", "rainfall", "soil_moisture"]
    env_inputs = {}
    for f in env_fields:
        raw = request.form.get(f, "").strip()
        if raw != "":
            try:
                env_inputs[f] = float(raw)
            except ValueError:
                return jsonify({"error": f"Invalid value for '{f}': must be a number"}), 400

    has_env_data = len(env_inputs) == len(env_fields)

    location = None
    if has_env_data:
        username = session.get("username")
        if not username:
            return jsonify({
                "error": "Please log in to submit weather data — this keeps your farm's "
                         "history separate from other users' data."
            }), 401
        user = auth_store.get_user(username)
        if user is None:
            session.pop("username", None)
            return jsonify({"error": "Your session has expired. Please log in again."}), 401
        location = user["farm_name"]

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1)[0]

        raw_probs = {CLASS_NAMES[i]: probs[i].item() for i in range(len(CLASS_NAMES))}

        if plant_type in PLANT_CLASSES:
            allowed = PLANT_CLASSES[plant_type]
            subset = {k: v for k, v in raw_probs.items() if k in allowed}
            total = sum(subset.values()) or 1e-9
            all_probs = {k: round((v / total) * 100, 2) for k, v in subset.items()}
            plant_filter_applied = plant_type
        else:
            all_probs = {k: round(v * 100, 2) for k, v in raw_probs.items()}

        top3 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top1_class = top3[0][0]
        top1_conf  = top3[0][1]

        # ── FR3 + FR4 — environmental risk and combined diagnosis ──────────
        env_result = None
        combined = None
        forecast = None
        if has_env_data:
            env_result = compute_current_dpis(
                temperature=env_inputs["temperature"],
                humidity=env_inputs["humidity"],
                rainfall=env_inputs["rainfall"],
                soil_moisture=env_inputs["soil_moisture"],
            )
            env_key = DISEASE_MAP.get(top1_class)
            combined = combine_diagnosis(top1_class, SEVERITY[top1_class], env_key, env_result)

            # Step 3 — record today's reading + attempt the 5-day forecast.
            # Isolated in its own try/except so a history/forecast problem
            # never breaks the core image + today's-DPI diagnosis above.
            try:
                dpis_only = {k: v["dpi"] for k, v in env_result.items()}
                history_store.upsert_today(
                    location, env_inputs["temperature"], env_inputs["humidity"],
                    env_inputs["rainfall"], env_inputs["soil_moisture"], dpis_only,
                )
                if env_key is not None:
                    forecast = build_forecast_block(location, env_key, env_result[env_key]["dpi"])
            except Exception as forecast_err:
                forecast = {"status": "unavailable", "message": f"Forecast error: {forecast_err}"}

        return jsonify({
            "predicted_class":      top1_class,
            "confidence":           top1_conf,
            "severity":             SEVERITY[top1_class],
            "treatment":            TREATMENT[top1_class],
            "top3":                 top3,
            "all_probs":            all_probs,
            "plant_filter_applied": plant_filter_applied,
            "environmental":        env_result,
            "combined_diagnosis":   combined,
            "forecast":             forecast,
            "location":             location,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    print("\n" + "="*50)
    print("  Plant Disease Detector running!")
    print("  Open your browser at: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)