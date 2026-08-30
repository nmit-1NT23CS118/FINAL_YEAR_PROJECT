(() => {
  "use strict";

  // ── friendly names, kept in sync with app.py's own dicts ──
  const SEVERITY = {
    "Pepper__bell___Bacterial_spot": "danger",
    "Pepper__bell___healthy": "healthy",
    "Potato___Early_blight": "warning",
    "Potato___Late_blight": "danger",
    "Potato___healthy": "healthy",
    "Tomato_Early_blight": "warning",
    "Tomato_Late_blight": "danger",
    "Tomato_healthy": "healthy",
  };

  const CLASS_DISPLAY = {
    "Pepper__bell___Bacterial_spot": "Pepper — Bacterial Spot",
    "Pepper__bell___healthy": "Pepper — Healthy",
    "Potato___Early_blight": "Potato — Early Blight",
    "Potato___Late_blight": "Potato — Late Blight",
    "Potato___healthy": "Potato — Healthy",
    "Tomato_Early_blight": "Tomato — Early Blight",
    "Tomato_Late_blight": "Tomato — Late Blight",
    "Tomato_healthy": "Tomato — Healthy",
  };

  const DISEASE_DISPLAY = {
    tomato_late_blight: "Tomato · Late Blight",
    potato_late_blight: "Potato · Late Blight",
    tomato_early_blight: "Tomato · Early Blight",
    potato_early_blight: "Potato · Early Blight",
    pepper_bacterial_spot: "Pepper · Bacterial Spot",
  };
  const RELEVANT_ENV_KEYS = Object.keys(DISEASE_DISPLAY);

  const ALERT_ICON = { healthy: "✓", warning: "!", danger: "⚠" };
  const ALERT_HEADLINE = {
    healthy: "Looking healthy",
    warning: "Early symptoms detected",
    danger: "Action needed",
  };

  // ── element refs ──
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const plantSelect = document.getElementById("plantSelect");

  const viewfinder = document.getElementById("viewfinder");
  const fileInput = document.getElementById("fileInput");
  const previewImg = document.getElementById("previewImg");
  const scanline = document.getElementById("scanline");

  const toggleReadings = document.getElementById("toggleReadings");
  const readingsPanel = document.getElementById("readingsPanel");

  const scanBtn = document.getElementById("scanBtn");
  const scanBtnLabel = document.getElementById("scanBtnLabel");
  const errorMsg = document.getElementById("errorMsg");

  const results = document.getElementById("results");
  const alertBanner = document.getElementById("alertBanner");
  const alertIcon = document.getElementById("alertIcon");
  const alertTitle = document.getElementById("alertTitle");
  const alertSub = document.getElementById("alertSub");
  const confidenceValue = document.getElementById("confidenceValue");

  const probList = document.getElementById("probList");
  const plantFilterNote = document.getElementById("plantFilterNote");
  const treatmentText = document.getElementById("treatmentText");

  const combinedCard = document.getElementById("combinedCard");
  const combinedMessage = document.getElementById("combinedMessage");

  const forecastCard = document.getElementById("forecastCard");
  const forecastBody = document.getElementById("forecastBody");

  const dpiSection = document.getElementById("dpiSection");
  const dpiGrid = document.getElementById("dpiGrid");

  // Auth elements
  const signInBtn = document.getElementById("signInBtn");
  const authUser = document.getElementById("authUser");
  const authFarmName = document.getElementById("authFarmName");
  const logoutBtn = document.getElementById("logoutBtn");
  const authModal = document.getElementById("authModal");
  const modalClose = document.getElementById("modalClose");
  const tabLogin = document.getElementById("tabLogin");
  const tabRegister = document.getElementById("tabRegister");
  const loginForm = document.getElementById("loginForm");
  const registerForm = document.getElementById("registerForm");
  const modalError = document.getElementById("modalError");
  const readingsAuthGate = document.getElementById("readingsAuthGate");
  const readingsLoggedIn = document.getElementById("readingsLoggedIn");
  const readingsLoggedInAs = document.getElementById("readingsLoggedInAs");
  const readingsSignInBtn = document.getElementById("readingsSignInBtn");

  let selectedFile = null;
  let currentUser = null; // { username, farm_name } or null

  // ── auth: session state ──
  function openModal(tab) {
    modalError.hidden = true;
    loginForm.reset();
    registerForm.reset();
    switchTab(tab || "login");
    authModal.hidden = false;
  }
  function closeModal() { authModal.hidden = true; }

  function switchTab(tab) {
    const isLogin = tab === "login";
    tabLogin.classList.toggle("active", isLogin);
    tabRegister.classList.toggle("active", !isLogin);
    loginForm.hidden = !isLogin;
    registerForm.hidden = isLogin;
    modalError.hidden = true;
  }

  signInBtn.addEventListener("click", () => openModal("login"));
  readingsSignInBtn.addEventListener("click", () => openModal("login"));
  modalClose.addEventListener("click", closeModal);
  authModal.addEventListener("click", e => { if (e.target === authModal) closeModal(); });
  tabLogin.addEventListener("click", () => switchTab("login"));
  tabRegister.addEventListener("click", () => switchTab("register"));

  function renderAuthState() {
    if (currentUser) {
      signInBtn.hidden = true;
      authUser.hidden = false;
      authFarmName.textContent = currentUser.farm_name;
      readingsAuthGate.hidden = true;
      readingsLoggedIn.hidden = false;
      readingsLoggedInAs.textContent = `LOGGING READINGS FOR: ${currentUser.farm_name.toUpperCase()}`;
    } else {
      signInBtn.hidden = false;
      authUser.hidden = true;
      readingsAuthGate.hidden = false;
      readingsLoggedIn.hidden = true;
    }
  }

  async function refreshAuth() {
    try {
      const res = await fetch("/api/me");
      const data = await res.json();
      currentUser = data.logged_in ? { username: data.username, farm_name: data.farm_name } : null;
    } catch {
      currentUser = null;
    }
    renderAuthState();
  }

  loginForm.addEventListener("submit", async e => {
    e.preventDefault();
    modalError.hidden = true;
    const username = document.getElementById("loginUsername").value.trim();
    const password = document.getElementById("loginPassword").value;
    try {
      const res = await fetch("/api/login", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Login failed.");
      currentUser = { username: data.username, farm_name: data.farm_name };
      renderAuthState();
      closeModal();
    } catch (err) {
      modalError.textContent = err.message;
      modalError.hidden = false;
    }
  });

  registerForm.addEventListener("submit", async e => {
    e.preventDefault();
    modalError.hidden = true;
    const username = document.getElementById("regUsername").value.trim();
    const farm_name = document.getElementById("regFarmName").value.trim();
    const password = document.getElementById("regPassword").value;
    try {
      const res = await fetch("/api/register", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password, farm_name }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Registration failed.");
      currentUser = { username: data.username, farm_name: data.farm_name };
      renderAuthState();
      closeModal();
    } catch (err) {
      modalError.textContent = err.message;
      modalError.hidden = false;
    }
  });

  logoutBtn.addEventListener("click", async () => {
    try { await fetch("/api/logout", { method: "POST" }); } catch {}
    currentUser = null;
    renderAuthState();
  });

  refreshAuth();

  // ── backend liveness + plant list ──
  fetch("/plants")
    .then(r => { if (!r.ok) throw new Error(); return r.json(); })
    .then(data => {
      statusDot.classList.add("ok");
      statusText.textContent = "MODEL READY";
      (data.plants || []).forEach(p => {
        const opt = document.createElement("option");
        opt.value = p; opt.textContent = p;
        plantSelect.appendChild(opt);
      });
    })
    .catch(() => {
      statusDot.classList.add("err");
      statusText.textContent = "SERVER UNREACHABLE";
    });

  // ── viewfinder interactions ──
  viewfinder.addEventListener("click", () => fileInput.click());
  viewfinder.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) setFile(fileInput.files[0]);
  });
  ["dragover", "dragenter"].forEach(evt =>
    viewfinder.addEventListener(evt, e => { e.preventDefault(); viewfinder.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach(evt =>
    viewfinder.addEventListener(evt, e => { e.preventDefault(); viewfinder.classList.remove("dragover"); })
  );
  viewfinder.addEventListener("drop", e => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) setFile(f);
  });

  function setFile(file) {
    selectedFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    viewfinder.classList.add("has-image");
    scanBtn.disabled = false;
    hideError();
  }

  // ── field readings toggle ──
  toggleReadings.addEventListener("click", () => {
    const open = readingsPanel.hidden;
    readingsPanel.hidden = !open;
    toggleReadings.setAttribute("aria-expanded", String(open));
    toggleReadings.querySelector(".toggle-icon").textContent = open ? "–" : "+";
  });

  // ── submit ──
  scanBtn.addEventListener("click", runScan);

  async function runScan() {
    if (!selectedFile) return;
    hideError();

    const fields = { temperature: "rTemp", humidity: "rHum", rainfall: "rRain", soil_moisture: "rSoil" };
    const rawValues = {};
    for (const [key, id] of Object.entries(fields)) {
      const el = document.getElementById(id);
      rawValues[key] = el ? el.value.trim() : "";
    }
    const anyFilled = Object.values(rawValues).some(v => v !== "");
    const allFilled = Object.values(rawValues).every(v => v !== "");
    if (anyFilled && !allFilled) {
      showError("Fill in all four field readings, or leave all four blank.");
      return;
    }
    if (allFilled && !currentUser) {
      showError("Please log in to submit weather readings.");
      openModal("login");
      return;
    }

    const form = new FormData();
    form.append("image", selectedFile);
    if (plantSelect.value) form.append("plant_type", plantSelect.value);
    if (allFilled) {
      for (const [key, id] of Object.entries(fields)) {
        form.append(key, document.getElementById(id).value.trim());
      }
    }

    setLoading(true);
    try {
      const res = await fetch("/predict", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "The scan failed. Try another image.");
      renderResults(data);
    } catch (err) {
      showError(err.message || "Something went wrong reaching the model.");
    } finally {
      setLoading(false);
    }
  }

  function setLoading(loading) {
    scanBtn.disabled = loading || !selectedFile;
    scanBtn.classList.toggle("loading", loading);
    scanBtnLabel.textContent = loading ? "Reading tissue…" : "Scan leaf";
    scanline.classList.toggle("active", loading);
  }

  function showError(msg) { errorMsg.textContent = msg; errorMsg.hidden = false; }
  function hideError() { errorMsg.hidden = true; }

  // ── render ──
  function renderResults(data) {
    results.hidden = false;

    // banner
    const sev = data.severity || "healthy";
    alertBanner.className = "alert-banner risk-" + sev;
    alertIcon.textContent = ALERT_ICON[sev] || "•";
    alertTitle.textContent = ALERT_HEADLINE[sev] || "Result";
    alertSub.textContent = CLASS_DISPLAY[data.predicted_class] || data.predicted_class;
    confidenceValue.textContent = data.confidence.toFixed(1) + "%";

    // top-3 bars
    probList.innerHTML = "";
    (data.top3 || []).forEach(([cls, val]) => {
      const row = document.createElement("div");
      row.className = "prob-row";
      const barColor = SEVERITY[cls] === "danger" ? "var(--red)" : SEVERITY[cls] === "warning" ? "var(--amber)" : "var(--leaf)";
      row.innerHTML = `
        <div class="prob-row-top">
          <span class="prob-name">${CLASS_DISPLAY[cls] || cls}</span>
          <span class="prob-value mono">${val.toFixed(1)}%</span>
        </div>
        <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${val}%; background:${barColor}"></div></div>
      `;
      probList.appendChild(row);
    });

    if (data.plant_filter_applied) {
      plantFilterNote.hidden = false;
      plantFilterNote.textContent = `FILTERED TO: ${data.plant_filter_applied.toUpperCase()}`;
    } else {
      plantFilterNote.hidden = true;
    }

    // treatment
    treatmentText.textContent = data.treatment || "—";

    // combined diagnosis
    if (data.combined_diagnosis && data.combined_diagnosis.available) {
      combinedCard.hidden = false;
      combinedMessage.textContent = data.combined_diagnosis.message;
    } else {
      combinedCard.hidden = true;
    }

    // forecast
    renderForecast(data.forecast);

    // DPI grid
    renderDpiGrid(data.environmental, data.combined_diagnosis);

    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderForecast(forecast) {
    if (!forecast) { forecastCard.hidden = true; return; }
    forecastCard.hidden = false;

    if (forecast.status === "ready") {
      const trendClass = "trend-" + forecast.trend;
      const arrow = forecast.trend === "rising" ? "↑" : forecast.trend === "falling" ? "↓" : "→";
      forecastBody.innerHTML = `
        <div class="forecast-ready">
          <div class="forecast-metric">
            <span class="forecast-metric-label mono">TODAY</span>
            <span class="forecast-metric-value mono">${forecast.current_dpi.toFixed(1)}</span>
          </div>
          <span class="forecast-arrow ${trendClass}">${arrow}</span>
          <div class="forecast-metric">
            <span class="forecast-metric-label mono">IN 5 DAYS</span>
            <span class="forecast-metric-value mono ${trendClass}">${forecast.forecast_dpi.toFixed(1)}</span>
          </div>
        </div>
        <p class="forecast-text">${forecast.message}</p>
      `;
    } else if (forecast.status === "collecting") {
      const pct = Math.round((forecast.days_recorded / forecast.days_needed) * 100);
      forecastBody.innerHTML = `
        <div class="forecast-status">
          <span class="mono">${forecast.days_recorded}/${forecast.days_needed} days</span>
          <div class="forecast-progress-track"><div class="forecast-progress-fill" style="width:${pct}%"></div></div>
        </div>
        <p class="forecast-text">${forecast.message}</p>
      `;
    } else {
      forecastBody.innerHTML = `<p class="forecast-text">${forecast.message}</p>`;
    }
  }

  function renderDpiGrid(envResult, combined) {
    if (!envResult) { dpiSection.hidden = true; return; }
    dpiSection.hidden = false;
    dpiGrid.innerHTML = "";

    const highlightKey = combined && combined.available ? combined.disease_key : null;

    RELEVANT_ENV_KEYS.forEach(key => {
      const entry = envResult[key];
      if (!entry) return;
      const card = document.createElement("div");
      card.className = `dpi-card risk-${entry.risk}` + (key === highlightKey ? " highlighted" : "");
      card.innerHTML = `
        <p class="dpi-card-name">${DISEASE_DISPLAY[key]}</p>
        <p class="dpi-card-value">${entry.dpi.toFixed(1)}</p>
        <span class="dpi-card-label mono">${entry.risk.toUpperCase()} RISK</span>
      `;
      dpiGrid.appendChild(card);
    });
  }
})();