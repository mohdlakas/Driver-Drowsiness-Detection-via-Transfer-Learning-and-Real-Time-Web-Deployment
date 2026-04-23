const els = {
  driverName: document.getElementById("driverName"),
  cameraSelect: document.getElementById("cameraSelect"),
  refreshCamerasBtn: document.getElementById("refreshCamerasBtn"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  thresholdSlider: document.getElementById("thresholdSlider"),
  thresholdValue: document.getElementById("thresholdValue"),
  intervalSlider: document.getElementById("intervalSlider"),
  intervalValue: document.getElementById("intervalValue"),
  historyIntervalSlider: document.getElementById("historyIntervalSlider"),
  historyIntervalValue: document.getElementById("historyIntervalValue"),
  audioToggle: document.getElementById("audioToggle"),
  realtimeToggle: document.getElementById("realtimeToggle"),
  preferIntegratedToggle: document.getElementById("preferIntegratedToggle"),
  backendStatus: document.getElementById("backendStatus"),
  deviceStatus: document.getElementById("deviceStatus"),
  modelStatus: document.getElementById("modelStatus"),
  video: document.getElementById("video"),
  hiddenCanvas: document.getElementById("hiddenCanvas"),
  videoBadge: document.getElementById("videoBadge"),
  predictionText: document.getElementById("predictionText"),
  confidenceText: document.getElementById("confidenceText"),
  latencyText: document.getElementById("latencyText"),
  drowsyBar: document.getElementById("drowsyBar"),
  nonDrowsyBar: document.getElementById("nonDrowsyBar"),
  drowsyText: document.getElementById("drowsyText"),
  nonDrowsyText: document.getElementById("nonDrowsyText"),
  alertText: document.getElementById("alertText"),
  uploadInput: document.getElementById("uploadInput"),
  analyzeUploadBtn: document.getElementById("analyzeUploadBtn"),
  uploadPreview: document.getElementById("uploadPreview"),
  historyBody: document.getElementById("historyBody"),
  downloadCsvBtn: document.getElementById("downloadCsvBtn"),
  clearHistoryBtn: document.getElementById("clearHistoryBtn"),
};

const state = {
  stream: null,
  inferTimer: null,
  inFlight: false,
  history: [],
  lastAlert: false,
  lastAlertTime: 0,
  lastHistoryLogTime: 0,
  audioCtx: null,
};

const ALERT_COOLDOWN_MS = 3000;
const MAX_HISTORY = 500;

function setBadgeLive(isLive) {
  els.videoBadge.textContent = isLive ? "LIVE" : "IDLE";
  els.videoBadge.className = `video-badge ${isLive ? "live" : "idle"}`;
}

function updateSliderLabels() {
  els.thresholdValue.textContent = Number(els.thresholdSlider.value).toFixed(2);
  els.intervalValue.textContent = String(parseInt(els.intervalSlider.value, 10));
  els.historyIntervalValue.textContent = Number(els.historyIntervalSlider.value).toFixed(1);
}

function choosePreferredCamera(videos) {
  if (videos.length === 0) {
    return "";
  }

  if (!els.preferIntegratedToggle.checked) {
    return videos[0].deviceId;
  }

  const integratedPattern = /(integrated|built[- ]?in|internal|webcam)/i;
  const virtualPattern = /(ivcam|e2esoft|obs|virtual|manycam|droidcam|snap camera)/i;

  const labeled = videos.filter((d) => d.label && d.label.trim().length > 0);
  const integrated = labeled.find((d) => integratedPattern.test(d.label) && !virtualPattern.test(d.label));
  if (integrated) {
    return integrated.deviceId;
  }

  const nonVirtual = labeled.find((d) => !virtualPattern.test(d.label));
  if (nonVirtual) {
    return nonVirtual.deviceId;
  }

  return videos[0].deviceId;
}

async function refreshCameraList(keepSelection = true) {
  if (!navigator.mediaDevices?.enumerateDevices) {
    els.cameraSelect.innerHTML = "<option>Camera API unavailable</option>";
    return;
  }

  const previous = els.cameraSelect.value;
  const devices = await navigator.mediaDevices.enumerateDevices();
  const videos = devices.filter((d) => d.kind === "videoinput");

  els.cameraSelect.innerHTML = "";
  videos.forEach((device, i) => {
    const label = device.label || `Camera ${i + 1}`;
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = label;
    els.cameraSelect.appendChild(option);
  });

  if (videos.length === 0) {
    const option = document.createElement("option");
    option.textContent = "No camera found";
    option.value = "";
    els.cameraSelect.appendChild(option);
    return;
  }

  const hasPrevious = videos.some((d) => d.deviceId === previous);
  if (keepSelection && hasPrevious) {
    els.cameraSelect.value = previous;
  } else {
    els.cameraSelect.value = choosePreferredCamera(videos);
  }
}

function stopCurrentStream() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  els.video.srcObject = null;
  setBadgeLive(false);
}

function stopInferenceLoop() {
  if (state.inferTimer) {
    clearInterval(state.inferTimer);
    state.inferTimer = null;
  }
}

function startInferenceLoop() {
  stopInferenceLoop();
  const intervalMs = parseInt(els.intervalSlider.value, 10);
  if (!els.realtimeToggle.checked) {
    return;
  }

  state.inferTimer = setInterval(() => {
    captureAndInfer().catch((err) => {
      console.error("Inference loop error:", err);
    });
  }, intervalMs);
}

async function startCamera() {
  stopCurrentStream();
  stopInferenceLoop();

  const selectedId = els.cameraSelect.value;
  let constraints = { video: true, audio: false };
  if (selectedId) {
    constraints = {
      video: {
        deviceId: { exact: selectedId },
        facingMode: "user",
      },
      audio: false,
    };
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.stream = stream;
    els.video.srcObject = stream;
    await els.video.play();
    setBadgeLive(true);
    await refreshCameraList(true);
    startInferenceLoop();
  } catch (exactErr) {
    console.warn("Exact device request failed, retrying fallback:", exactErr);
    try {
      const fallback = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      state.stream = fallback;
      els.video.srcObject = fallback;
      await els.video.play();
      setBadgeLive(true);
      await refreshCameraList(true);
      startInferenceLoop();
    } catch (err) {
      console.error("Camera start failed:", err);
      alert(`Camera access failed: ${err.message || err}`);
    }
  }
}

async function blobFromVideoFrame() {
  const video = els.video;
  if (!video.videoWidth || !video.videoHeight) {
    return null;
  }

  const maxWidth = 640;
  const scale = maxWidth / video.videoWidth;
  const targetWidth = maxWidth;
  const targetHeight = Math.max(1, Math.floor(video.videoHeight * scale));

  const canvas = els.hiddenCanvas;
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, targetWidth, targetHeight);

  return await new Promise((resolve) => {
    canvas.toBlob(resolve, "image/jpeg", 0.85);
  });
}

function addHistory(source, result) {
  const nowMs = Date.now();
  if (source === "Realtime") {
    const minGapMs = Number(els.historyIntervalSlider.value) * 1000;
    if (nowMs - state.lastHistoryLogTime < minGapMs) {
      return;
    }
    state.lastHistoryLogTime = nowMs;
  }

  const row = {
    time: new Date().toLocaleString(),
    driver: (els.driverName.value || "Unknown").trim(),
    source,
    prediction: result.prediction_label,
    confidence: result.confidence,
    drowsy: result.drowsy_prob,
    nonDrowsy: result.non_drowsy_prob,
    latency: result.latency_ms,
  };

  state.history.unshift(row);
  if (state.history.length > MAX_HISTORY) {
    state.history.pop();
  }
  renderHistory();
}

function renderHistory() {
  if (state.history.length === 0) {
    els.historyBody.innerHTML = "<tr><td colspan='7'>No records yet.</td></tr>";
    return;
  }

  els.historyBody.innerHTML = state.history
    .map(
      (row) => `
        <tr>
          <td>${row.time}</td>
          <td>${row.driver}</td>
          <td>${row.source}</td>
          <td>${row.prediction}</td>
          <td>${(row.confidence * 100).toFixed(1)}%</td>
          <td>${(row.drowsy * 100).toFixed(1)}%</td>
          <td>${row.latency.toFixed(1)} ms</td>
        </tr>
      `
    )
    .join("");
}

function playAlertTone() {
  const now = Date.now();
  if (now - state.lastAlertTime < ALERT_COOLDOWN_MS) {
    return;
  }

  if (!state.audioCtx) {
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  state.audioCtx.resume();

  const osc = state.audioCtx.createOscillator();
  const gain = state.audioCtx.createGain();
  osc.type = "sine";
  osc.frequency.value = 920;
  gain.gain.value = 0.0001;
  osc.connect(gain);
  gain.connect(state.audioCtx.destination);

  const t = state.audioCtx.currentTime;
  gain.gain.exponentialRampToValueAtTime(0.2, t + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.28);
  osc.start(t);
  osc.stop(t + 0.29);
  state.lastAlertTime = now;
}

function updateResultView(result) {
  const drowsy = Number(result.drowsy_prob);
  const nonDrowsy = Number(result.non_drowsy_prob);
  const threshold = Number(els.thresholdSlider.value);
  const isAlert = drowsy >= threshold;

  els.predictionText.textContent = result.prediction_label;
  els.confidenceText.textContent = `${(Number(result.confidence) * 100).toFixed(2)}%`;
  els.latencyText.textContent = `${Number(result.latency_ms).toFixed(1)} ms`;

  els.drowsyBar.style.width = `${Math.max(0, Math.min(100, drowsy * 100))}%`;
  els.nonDrowsyBar.style.width = `${Math.max(0, Math.min(100, nonDrowsy * 100))}%`;
  els.drowsyText.textContent = `${(drowsy * 100).toFixed(1)}%`;
  els.nonDrowsyText.textContent = `${(nonDrowsy * 100).toFixed(1)}%`;

  els.alertText.className = `alert-text ${isAlert ? "risk" : "safe"}`;
  els.alertText.textContent = isAlert
    ? "High risk detected: drowsy threshold crossed."
    : "No high-risk drowsiness signal.";

  if (els.audioToggle.checked && isAlert && !state.lastAlert) {
    playAlertTone();
  }
  state.lastAlert = isAlert;
}

async function sendForInference(blob, source) {
  if (state.inFlight || !blob) {
    return;
  }
  state.inFlight = true;

  try {
    const formData = new FormData();
    formData.append("frame", blob, "frame.jpg");

    const resp = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Inference failed: ${resp.status} ${text}`);
    }

    const result = await resp.json();
    updateResultView(result);
    addHistory(source, result);
  } catch (err) {
    console.error(err);
    els.alertText.className = "alert-text risk";
    els.alertText.textContent = `Inference error: ${err.message || err}`;
  } finally {
    state.inFlight = false;
  }
}

async function captureAndInfer() {
  if (!state.stream || !els.realtimeToggle.checked) {
    return;
  }

  if (els.video.readyState < 2) {
    return;
  }

  const blob = await blobFromVideoFrame();
  if (!blob) {
    return;
  }
  await sendForInference(blob, "Realtime");
}

function downloadCsv() {
  const headers = ["Time", "Driver", "Source", "Prediction", "Confidence", "Drowsy Prob", "Non Drowsy Prob", "Latency (ms)"];
  const lines = [headers.join(",")];

  state.history.forEach((row) => {
    lines.push(
      [
        row.time,
        row.driver,
        row.source,
        row.prediction,
        row.confidence.toFixed(6),
        row.drowsy.toFixed(6),
        row.nonDrowsy.toFixed(6),
        row.latency.toFixed(2),
      ]
        .map((cell) => `"${String(cell).replaceAll('"', '""')}"`)
        .join(",")
    );
  });

  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `drowsiness_session_${new Date().toISOString().replaceAll(":", "-")}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function checkBackend() {
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    els.backendStatus.textContent = data.status;
    els.deviceStatus.textContent = data.device || "-";
    els.modelStatus.textContent = data.model_path || "-";

    if (data.status !== "ok") {
      els.alertText.className = "alert-text risk";
      els.alertText.textContent = `Backend/model issue: ${data.message || "unknown error"}`;
    }
  } catch (err) {
    els.backendStatus.textContent = "error";
    els.alertText.className = "alert-text risk";
    els.alertText.textContent = `Cannot reach backend: ${err.message || err}`;
  }
}

function wireEvents() {
  els.refreshCamerasBtn.addEventListener("click", () => refreshCameraList(false));
  els.startBtn.addEventListener("click", () => startCamera());
  els.stopBtn.addEventListener("click", () => {
    stopInferenceLoop();
    stopCurrentStream();
    state.lastAlert = false;
  });

  els.intervalSlider.addEventListener("input", () => {
    updateSliderLabels();
    startInferenceLoop();
  });
  els.thresholdSlider.addEventListener("input", updateSliderLabels);
  els.historyIntervalSlider.addEventListener("input", updateSliderLabels);
  els.realtimeToggle.addEventListener("change", startInferenceLoop);
  els.preferIntegratedToggle.addEventListener("change", () => refreshCameraList(false));

  els.uploadInput.addEventListener("change", () => {
    const file = els.uploadInput.files?.[0];
    if (!file) {
      els.uploadPreview.style.display = "none";
      return;
    }
    const url = URL.createObjectURL(file);
    els.uploadPreview.src = url;
    els.uploadPreview.style.display = "block";
  });

  els.analyzeUploadBtn.addEventListener("click", async () => {
    const file = els.uploadInput.files?.[0];
    if (!file) {
      alert("Select an image file first.");
      return;
    }
    await sendForInference(file, "Upload");
  });

  els.downloadCsvBtn.addEventListener("click", downloadCsv);
  els.clearHistoryBtn.addEventListener("click", () => {
    state.history = [];
    renderHistory();
  });

  els.cameraSelect.addEventListener("change", async () => {
    if (!state.stream) {
      return;
    }
    await startCamera();
  });
}

async function init() {
  updateSliderLabels();
  renderHistory();
  wireEvents();
  await checkBackend();
  await refreshCameraList(false);

  if (navigator.mediaDevices?.addEventListener) {
    navigator.mediaDevices.addEventListener("devicechange", async () => {
      await refreshCameraList(true);
    });
  }
}

init().catch((err) => {
  console.error("Init failed:", err);
  els.alertText.className = "alert-text risk";
  els.alertText.textContent = `Initialization failed: ${err.message || err}`;
});
