import { RppgBuffer, estimateHeartRate } from "./rppg.js";
import { clearCharts, drawSignalChart, drawSpectrumChart } from "./charts.js";

const MEDIAPIPE_VERSION = "1.0.1";
const MEDIAPIPE_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const FACE_DETECTION_INTERVAL_MS = 80;
const MAX_FACE_AGE_MS = 420;
const ANALYSIS_INTERVAL_MS = 700;
const DISPLAY_CONFIDENCE_THRESHOLD = 0.42;
const SAMPLE_LONG_EDGE = 320;

const elements = {
  video: document.querySelector("#camera"),
  overlay: document.querySelector("#overlay"),
  cameraStage: document.querySelector("#cameraStage"),
  cameraPlaceholder: document.querySelector("#cameraPlaceholder"),
  cameraCoach: document.querySelector("#cameraCoach"),
  coachText: document.querySelector("#coachText"),
  liveState: document.querySelector("#liveState"),
  startButton: document.querySelector("#startButton"),
  stopButton: document.querySelector("#stopButton"),
  resetButton: document.querySelector("#resetButton"),
  fullscreenButton: document.querySelector("#fullscreenButton"),
  cameraSelect: document.querySelector("#cameraSelect"),
  minBpm: document.querySelector("#minBpm"),
  maxBpm: document.querySelector("#maxBpm"),
  windowSeconds: document.querySelector("#windowSeconds"),
  mirrorToggle: document.querySelector("#mirrorToggle"),
  roiToggle: document.querySelector("#roiToggle"),
  exportButton: document.querySelector("#exportButton"),
  bpmDisplay: document.querySelector("#bpmDisplay"),
  bpmValue: document.querySelector("#bpmValue"),
  pulseHalo: document.querySelector("#pulseHalo"),
  confidenceValue: document.querySelector("#confidenceValue"),
  confidenceBar: document.querySelector("#confidenceBar"),
  collectionLabel: document.querySelector("#collectionLabel"),
  collectionTime: document.querySelector("#collectionTime"),
  collectionBar: document.querySelector("#collectionBar"),
  qualityFace: document.querySelector("#qualityFace"),
  qualityLight: document.querySelector("#qualityLight"),
  qualityMotion: document.querySelector("#qualityMotion"),
  qualitySignal: document.querySelector("#qualitySignal"),
  averageBpm: document.querySelector("#averageBpm"),
  rangeBpm: document.querySelector("#rangeBpm"),
  cameraFps: document.querySelector("#cameraFps"),
  signalChart: document.querySelector("#signalChart"),
  spectrumChart: document.querySelector("#spectrumChart"),
  peakLabel: document.querySelector("#peakLabel"),
};

const overlayContext = elements.overlay.getContext("2d");
const sampleCanvas = document.createElement("canvas");
const sampleContext = sampleCanvas.getContext("2d", { willReadFrequently: true });
const buffer = new RppgBuffer(32);

const state = {
  running: false,
  starting: false,
  stream: null,
  landmarker: null,
  landmarkerPromise: null,
  frameRequest: null,
  animationRequest: null,
  wakeLock: null,
  lastDetectionAt: -Infinity,
  lastAnalysisAt: -Infinity,
  lastFallbackVideoTime: -1,
  geometry: null,
  previousGeometry: null,
  motionScore: 1,
  frameQuality: null,
  currentEstimate: null,
  displayedBpm: null,
  rawEstimateHistory: [],
  sessionEstimates: [],
  frameTimes: [],
  detectorErrorCount: 0,
  lastSampleAt: -Infinity,
};

initialiseUi();

function initialiseUi() {
  clearCharts(elements.signalChart, elements.spectrumChart);
  elements.cameraStage.classList.toggle("is-mirrored", elements.mirrorToggle.checked);
  elements.startButton.addEventListener("click", () => startMeasurement());
  elements.stopButton.addEventListener("click", () => stopMeasurement({ keepResult: true }));
  elements.resetButton.addEventListener("click", resetSession);
  elements.fullscreenButton.addEventListener("click", toggleFullscreen);
  elements.cameraSelect.addEventListener("change", async () => {
    if (!state.running) return;
    await stopMeasurement({ keepResult: false, keepModel: true });
    await startMeasurement();
  });
  elements.mirrorToggle.addEventListener("change", () => {
    elements.cameraStage.classList.toggle("is-mirrored", elements.mirrorToggle.checked);
  });
  elements.roiToggle.addEventListener("change", drawOverlay);
  elements.minBpm.addEventListener("change", validateBpmRange);
  elements.maxBpm.addEventListener("change", validateBpmRange);
  elements.windowSeconds.addEventListener("change", () => {
    buffer.clear();
    clearAnalysisOnly();
  });
  elements.exportButton.addEventListener("click", exportCsv);
  document.addEventListener("visibilitychange", handleVisibilityChange);
  window.addEventListener("resize", redrawCharts);
  window.addEventListener("beforeunload", () => stopTracks());
}

async function startMeasurement() {
  if (state.running || state.starting) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    showFatalError("This browser does not provide camera access.");
    return;
  }
  if (!window.isSecureContext) {
    showFatalError("Camera access needs HTTPS. Open the GitHub Pages address using https://.");
    return;
  }

  state.starting = true;
  resetSession();
  setControlsForStarting();
  setLiveState("idle", "Starting");
  setCoach("Loading face tracker and requesting camera…", "warn");

  try {
    const [, stream] = await Promise.all([initialiseFaceLandmarker(), openCamera()]);
    state.stream = stream;
    state.running = true;
    state.starting = false;
    state.detectorErrorCount = 0;
    elements.cameraStage.classList.add("is-live");
    elements.startButton.disabled = true;
    elements.stopButton.disabled = false;
    elements.resetButton.disabled = false;
    elements.cameraSelect.disabled = false;
    setLiveState("live", "Live");
    setCoach("Centre your face and keep still", "warn");
    await requestWakeLock();
    scheduleNextFrame();
  } catch (error) {
    state.starting = false;
    stopTracks();
    setControlsForStopped();
    showFatalError(describeCameraError(error));
    console.error(error);
  }
}

async function initialiseFaceLandmarker() {
  if (state.landmarker) return state.landmarker;
  if (state.landmarkerPromise) return state.landmarkerPromise;

  state.landmarkerPromise = (async () => {
    const visionTasks = await import(`${MEDIAPIPE_ROOT}/vision_bundle.mjs`);
    const vision = await visionTasks.FilesetResolver.forVisionTasks(`${MEDIAPIPE_ROOT}/wasm`);
    const commonOptions = {
      runningMode: "VIDEO",
      numFaces: 1,
      minFaceDetectionConfidence: 0.55,
      minFacePresenceConfidence: 0.55,
      minTrackingConfidence: 0.55,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    };

    try {
      state.landmarker = await visionTasks.FaceLandmarker.createFromOptions(vision, {
        ...commonOptions,
        baseOptions: { modelAssetPath: FACE_MODEL_URL, delegate: "GPU" },
      });
    } catch (gpuError) {
      console.warn("GPU face tracking was unavailable; using CPU.", gpuError);
      state.landmarker = await visionTasks.FaceLandmarker.createFromOptions(vision, {
        ...commonOptions,
        baseOptions: { modelAssetPath: FACE_MODEL_URL, delegate: "CPU" },
      });
    }
    return state.landmarker;
  })();

  try {
    return await state.landmarkerPromise;
  } catch (error) {
    state.landmarkerPromise = null;
    throw new Error("The face-tracking model could not be loaded. Check the connection and reload the page.", { cause: error });
  }
}

async function openCamera() {
  stopTracks();
  const selectedDevice = elements.cameraSelect.value;
  const videoConstraints = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 30, min: 15 },
    facingMode: selectedDevice ? undefined : { ideal: "user" },
    deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
  };

  const stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
  elements.video.srcObject = stream;
  await new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => reject(new Error("The camera did not start in time.")), 12000);
    elements.video.onloadedmetadata = () => {
      window.clearTimeout(timeout);
      resolve();
    };
  });
  await elements.video.play();
  configureCanvases();
  await refreshCameraList(stream);
  return stream;
}

async function refreshCameraList(stream) {
  const currentTrack = stream.getVideoTracks()[0];
  const currentDeviceId = currentTrack?.getSettings?.().deviceId || elements.cameraSelect.value;
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === "videoinput");
  elements.cameraSelect.replaceChildren();
  devices.forEach((device, index) => {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `Camera ${index + 1}`;
    option.selected = device.deviceId === currentDeviceId;
    elements.cameraSelect.append(option);
  });
  if (devices.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Default camera";
    elements.cameraSelect.append(option);
  }
}

function configureCanvases() {
  const width = elements.video.videoWidth || 1280;
  const height = elements.video.videoHeight || 720;
  elements.overlay.width = width;
  elements.overlay.height = height;
  elements.cameraStage.style.setProperty("--camera-ratio", `${width} / ${height}`);
  const scale = SAMPLE_LONG_EDGE / Math.max(width, height);
  sampleCanvas.width = Math.max(1, Math.round(width * scale));
  sampleCanvas.height = Math.max(1, Math.round(height * scale));
}

function scheduleNextFrame() {
  if (!state.running) return;
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    state.frameRequest = elements.video.requestVideoFrameCallback(processVideoFrame);
  } else {
    state.animationRequest = requestAnimationFrame(processAnimationFrame);
  }
}

function processAnimationFrame(now) {
  if (!state.running) return;
  if (elements.video.currentTime !== state.lastFallbackVideoTime) {
    state.lastFallbackVideoTime = elements.video.currentTime;
    processVideoFrame(now, { mediaTime: elements.video.currentTime });
    return;
  }
  scheduleNextFrame();
}

function processVideoFrame(now) {
  if (!state.running) return;
  state.frameTimes.push(now);
  const frameCutoff = now - 3000;
  while (state.frameTimes[0] < frameCutoff) state.frameTimes.shift();

  if (now - state.lastDetectionAt >= FACE_DETECTION_INTERVAL_MS) {
    state.lastDetectionAt = now;
    detectFace(now);
  }

  const faceIsFresh = state.geometry && now - state.geometry.updatedAt <= MAX_FACE_AGE_MS;
  if (faceIsFresh) {
    const sampled = sampleSkinRegions(state.geometry.rois);
    if (sampled) {
      const lightScore = calculateLightScore(sampled);
      state.frameQuality = {
        faceScore: state.geometry.faceScore,
        poseScore: state.geometry.poseScore,
        motionScore: state.motionScore,
        lightScore,
        averageLuma: average(sampled.map((roi) => roi.luma)),
        clippedRatio: average(sampled.map((roi) => roi.clippedRatio)),
      };
      if (now - state.lastSampleAt > 850 && Number.isFinite(state.lastSampleAt)) {
        buffer.clear();
        clearAnalysisOnly();
        setCoach("Signal reset after face tracking was interrupted", "warn");
      }
      state.lastSampleAt = now;
      buffer.add(now / 1000, sampled, state.frameQuality);
      elements.exportButton.disabled = buffer.duration < 1;
    }
  } else {
    state.frameQuality = null;
  }

  updateLiveQuality(now);
  drawOverlay();

  if (now - state.lastAnalysisAt >= ANALYSIS_INTERVAL_MS) {
    state.lastAnalysisAt = now;
    runAnalysis();
  } else {
    updateCollectionProgress();
  }
  scheduleNextFrame();
}

function detectFace(now) {
  try {
    const result = state.landmarker.detectForVideo(elements.video, now);
    const landmarks = result.faceLandmarks?.[0];
    if (!landmarks || landmarks.length < 455) {
      if (state.geometry && now - state.geometry.updatedAt > MAX_FACE_AGE_MS) state.geometry = null;
      return;
    }
    state.detectorErrorCount = 0;
    updateGeometry(landmarks, now);
  } catch (error) {
    state.detectorErrorCount += 1;
    console.warn("Face tracking frame failed", error);
    if (state.detectorErrorCount >= 5) {
      showFatalError("Face tracking stopped unexpectedly. Stop and start the measurement again.");
      stopMeasurement({ keepResult: true });
    }
  }
}

function updateGeometry(landmarks, now) {
  const left = point(landmarks[234]);
  const right = point(landmarks[454]);
  const top = point(landmarks[10]);
  const chin = point(landmarks[152]);
  const nose = point(landmarks[1]);
  const horizontal = subtract(right, left);
  const vertical = subtract(chin, top);
  const faceWidth = magnitude(horizontal);
  const faceHeight = magnitude(vertical);
  if (faceWidth < 0.03 || faceHeight < 0.04) return;

  const ux = scalePoint(horizontal, 1 / faceWidth);
  const uy = scalePoint(vertical, 1 / faceHeight);
  const centre = midpoint(left, right);
  const nosePosition = dot(subtract(nose, left), ux) / faceWidth;
  const yawScore = clamp(1 - Math.abs(nosePosition - 0.5) / 0.27, 0, 1);
  const rollDegrees = Math.abs(Math.atan2(horizontal.y, horizontal.x) * 180 / Math.PI);
  const rollScore = clamp(1 - Math.max(0, rollDegrees - 7) / 32, 0, 1);
  const poseScore = Math.sqrt(yawScore * rollScore);
  const nearScore = clamp((faceWidth - 0.13) / 0.14, 0, 1);
  const farScore = clamp((0.9 - faceWidth) / 0.16, 0, 1);
  const faceScore = Math.sqrt(nearScore * farScore);

  const localPoint = (xFraction, yFraction) => {
    const centreLine = {
      x: top.x * (1 - yFraction) + chin.x * yFraction,
      y: top.y * (1 - yFraction) + chin.y * yFraction,
    };
    return add(centreLine, scalePoint(ux, faceWidth * xFraction));
  };

  const rois = [
    rotatedRectangle(localPoint(0, 0.22), ux, uy, faceWidth * 0.155, faceHeight * 0.055),
    rotatedRectangle(localPoint(-0.215, 0.51), ux, uy, faceWidth * 0.105, faceHeight * 0.072),
    rotatedRectangle(localPoint(0.215, 0.51), ux, uy, faceWidth * 0.105, faceHeight * 0.072),
  ].map((polygon) => polygon.map((p) => ({ x: clamp(p.x, 0, 1), y: clamp(p.y, 0, 1) })));

  const geometry = { landmarks, left, right, top, chin, centre, faceWidth, faceHeight, ux, uy, rois, poseScore, faceScore, updatedAt: now };
  if (state.previousGeometry) {
    const centreMotion = magnitude(subtract(centre, state.previousGeometry.centre)) / faceWidth;
    const scaleMotion = Math.abs(Math.log(faceWidth / state.previousGeometry.faceWidth));
    const instantMotion = centreMotion * 6 + scaleMotion * 2.5;
    const instantScore = Math.exp(-2 * instantMotion);
    state.motionScore = clamp(state.motionScore * 0.72 + instantScore * 0.28, 0, 1);
  } else {
    state.motionScore = 1;
  }
  state.previousGeometry = geometry;
  state.geometry = geometry;
}

function sampleSkinRegions(polygons) {
  if (!sampleCanvas.width || elements.video.readyState < 2) return null;
  sampleContext.drawImage(elements.video, 0, 0, sampleCanvas.width, sampleCanvas.height);
  const frame = sampleContext.getImageData(0, 0, sampleCanvas.width, sampleCanvas.height);
  const rois = polygons.map((polygon) => samplePolygon(frame, polygon));
  return rois.every((roi) => roi && roi.validRatio >= 0.55) ? rois : null;
}

function samplePolygon(frame, normalisedPolygon) {
  const width = frame.width;
  const height = frame.height;
  const polygon = normalisedPolygon.map((p) => ({ x: p.x * width, y: p.y * height }));
  const minX = clamp(Math.floor(Math.min(...polygon.map((p) => p.x))), 0, width - 1);
  const maxX = clamp(Math.ceil(Math.max(...polygon.map((p) => p.x))), 0, width - 1);
  const minY = clamp(Math.floor(Math.min(...polygon.map((p) => p.y))), 0, height - 1);
  const maxY = clamp(Math.ceil(Math.max(...polygon.map((p) => p.y))), 0, height - 1);
  const stride = Math.max(1, Math.round(Math.max(maxX - minX, maxY - minY) / 45));
  let total = 0;
  let valid = 0;
  let clipped = 0;
  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  let sumLuma = 0;

  for (let y = minY; y <= maxY; y += stride) {
    for (let x = minX; x <= maxX; x += stride) {
      if (!isPointInPolygon(x + 0.5, y + 0.5, polygon)) continue;
      total += 1;
      const offset = (y * width + x) * 4;
      const r = frame.data[offset];
      const g = frame.data[offset + 1];
      const b = frame.data[offset + 2];
      const minimum = Math.min(r, g, b);
      const maximum = Math.max(r, g, b);
      if (minimum <= 4 || maximum >= 251) {
        clipped += 1;
        continue;
      }
      const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      if (luma < 12 || luma > 245) continue;
      valid += 1;
      sumR += r;
      sumG += g;
      sumB += b;
      sumLuma += luma;
    }
  }

  if (valid < 20) return null;
  return {
    r: sumR / valid,
    g: sumG / valid,
    b: sumB / valid,
    luma: sumLuma / valid,
    validRatio: valid / Math.max(1, total),
    clippedRatio: clipped / Math.max(1, total),
  };
}

function calculateLightScore(rois) {
  const luma = average(rois.map((roi) => roi.luma));
  const clipped = average(rois.map((roi) => roi.clippedRatio));
  const darkScore = clamp((luma - 28) / 55, 0, 1);
  const brightScore = clamp((235 - luma) / 45, 0, 1);
  const clippingScore = clamp(1 - clipped / 0.12, 0, 1);
  return Math.cbrt(darkScore * brightScore * clippingScore);
}

function runAnalysis() {
  const options = getAnalysisOptions();
  const estimate = estimateHeartRate(buffer.recent(options.windowSeconds + 0.25), {
    ...options,
    previousBpm: state.displayedBpm,
  });
  state.currentEstimate = estimate;
  updateAnalysisUi(estimate);
}

function updateAnalysisUi(estimate) {
  const faceFresh = state.geometry && performance.now() - state.geometry.updatedAt <= MAX_FACE_AGE_MS * 2;
  const accepted = faceFresh && Number.isFinite(estimate.bpm) && estimate.confidence >= DISPLAY_CONFIDENCE_THRESHOLD;
  if (accepted) {
    state.rawEstimateHistory.push(estimate.bpm);
    if (state.rawEstimateHistory.length > 7) state.rawEstimateHistory.shift();
    const robustBpm = median(state.rawEstimateHistory);
    state.displayedBpm = Number.isFinite(state.displayedBpm)
      ? state.displayedBpm * 0.72 + robustBpm * 0.28
      : robustBpm;
    const rounded = Math.round(state.displayedBpm);
    elements.bpmValue.textContent = String(rounded);
    elements.bpmDisplay.dataset.ready = "true";
    elements.bpmDisplay.style.setProperty("--beat-duration", `${Math.round(60000 / state.displayedBpm)}ms`);

    const lastSession = state.sessionEstimates.at(-1);
    const now = performance.now();
    if (!lastSession || now - lastSession.time >= 900) {
      state.sessionEstimates.push({ time: now, bpm: state.displayedBpm, confidence: estimate.confidence });
      if (state.sessionEstimates.length > 600) state.sessionEstimates.shift();
      updateSessionSummary();
    }
  } else {
    elements.bpmValue.textContent = "—";
    elements.bpmDisplay.dataset.ready = "false";
  }

  const confidencePercent = Math.round(estimate.confidence * 100);
  elements.confidenceBar.setAttribute("aria-valuenow", String(confidencePercent));
  elements.confidenceBar.querySelector("span").style.width = `${confidencePercent}%`;
  elements.confidenceBar.dataset.level = estimate.confidence < 0.45 ? "low" : "good";
  elements.confidenceValue.textContent = confidenceLabel(estimate.confidence, estimate.duration);
  elements.cameraFps.textContent = estimate.fps > 0 ? `${estimate.fps.toFixed(0)} fps` : liveFpsLabel();
  setQuality(elements.qualitySignal, estimate.confidence >= 0.62 ? "good" : estimate.confidence >= 0.35 ? "warn" : "bad", signalQualityLabel(estimate));

  drawSignalChart(elements.signalChart, estimate.waveform);
  drawSpectrumChart(elements.spectrumChart, estimate.spectrum, accepted ? state.displayedBpm : estimate.bpm);
  elements.peakLabel.textContent = Number.isFinite(estimate.bpm) ? `${estimate.bpm.toFixed(1)} BPM peak` : "no peak";
  updateCollectionProgress(estimate);
}

function updateLiveQuality(now) {
  const geometry = state.geometry && now - state.geometry.updatedAt <= MAX_FACE_AGE_MS ? state.geometry : null;
  if (!geometry) {
    setQuality(elements.qualityFace, "bad", "Not found");
    setQuality(elements.qualityLight, "idle", "Waiting");
    setQuality(elements.qualityMotion, "idle", "Waiting");
    setCoach("Place one face inside the camera view", "bad");
    return;
  }

  const faceLevel = geometry.faceScore >= 0.7 && geometry.poseScore >= 0.65 ? "good" : geometry.faceScore >= 0.32 ? "warn" : "bad";
  const faceText = geometry.faceScore < 0.42 ? "Move closer" : geometry.poseScore < 0.62 ? "Face forward" : "Tracked";
  setQuality(elements.qualityFace, faceLevel, faceText);

  if (!state.frameQuality) {
    setQuality(elements.qualityLight, "bad", "Unreadable");
    setQuality(elements.qualityMotion, "warn", "Checking");
    setCoach("Keep forehead and cheeks unobstructed", "warn");
    return;
  }

  const lightLevel = state.frameQuality.lightScore >= 0.72 ? "good" : state.frameQuality.lightScore >= 0.4 ? "warn" : "bad";
  const lightText = state.frameQuality.averageLuma < 75 ? "Too dark" : state.frameQuality.averageLuma > 215 ? "Too bright" : state.frameQuality.clippedRatio > 0.08 ? "Glare" : "Good";
  setQuality(elements.qualityLight, lightLevel, lightText);

  const motionLevel = state.frameQuality.motionScore >= 0.72 ? "good" : state.frameQuality.motionScore >= 0.45 ? "warn" : "bad";
  setQuality(elements.qualityMotion, motionLevel, state.frameQuality.motionScore >= 0.72 ? "Still" : "Moving");

  const minimumDuration = Math.min(8, Number(elements.windowSeconds.value) - 2);
  if (geometry.faceScore < 0.42) setCoach("Move a little closer to the camera", "warn");
  else if (geometry.poseScore < 0.6) setCoach("Look straight toward the camera", "warn");
  else if (state.frameQuality.lightScore < 0.42) setCoach(lightText === "Too dark" ? "Add steady light in front of your face" : "Reduce glare or very bright light", "warn");
  else if (state.frameQuality.motionScore < 0.52) setCoach("Hold still — movement hides the colour signal", "warn");
  else if (buffer.duration < minimumDuration) setCoach("Good — keep still while the signal builds", "good");
  else if (state.currentEstimate?.confidence < DISPLAY_CONFIDENCE_THRESHOLD) setCoach(state.currentEstimate.diagnostics?.reasons?.[0] || "Keep still a little longer", "warn");
  else setCoach("Strong signal — measurement is live", "good");
}

function updateCollectionProgress(estimate = state.currentEstimate) {
  const target = Number(elements.windowSeconds.value);
  const duration = Math.min(buffer.duration, target);
  const progress = estimate?.progress ?? clamp(duration / target, 0, 1);
  elements.collectionBar.style.width = `${Math.round(progress * 100)}%`;
  elements.collectionTime.textContent = `${duration.toFixed(1)} s`;
  elements.collectionLabel.textContent = duration <= 0 ? "No signal collected" : progress < 0.95 ? "Building analysis window" : "Analysis window full";
}

function drawOverlay() {
  const width = elements.overlay.width;
  const height = elements.overlay.height;
  overlayContext.clearRect(0, 0, width, height);
  if (!state.running || !state.geometry) return;
  const geometry = state.geometry;
  const colour = state.frameQuality?.motionScore < 0.45 ? "#f4c96b" : "#55e6a5";

  overlayContext.save();
  overlayContext.lineWidth = Math.max(2, width / 650);
  overlayContext.strokeStyle = colour;
  overlayContext.globalAlpha = 0.76;
  overlayContext.setLineDash([width / 140, width / 170]);
  overlayContext.beginPath();
  overlayContext.ellipse(
    geometry.centre.x * width,
    (geometry.top.y + geometry.faceHeight * 0.5) * height,
    geometry.faceWidth * width * 0.57,
    geometry.faceHeight * height * 0.54,
    Math.atan2(geometry.ux.y, geometry.ux.x),
    0,
    Math.PI * 2,
  );
  overlayContext.stroke();
  overlayContext.setLineDash([]);

  if (elements.roiToggle.checked) {
    for (const polygon of geometry.rois) {
      overlayContext.beginPath();
      polygon.forEach((p, index) => {
        const x = p.x * width;
        const y = p.y * height;
        if (index === 0) overlayContext.moveTo(x, y);
        else overlayContext.lineTo(x, y);
      });
      overlayContext.closePath();
      overlayContext.fillStyle = "rgba(85, 230, 165, 0.14)";
      overlayContext.strokeStyle = colour;
      overlayContext.fill();
      overlayContext.stroke();
    }
  }
  overlayContext.restore();
}

async function stopMeasurement({ keepResult = true, keepModel = true } = {}) {
  state.running = false;
  state.starting = false;
  if (state.frameRequest != null && "cancelVideoFrameCallback" in HTMLVideoElement.prototype) {
    elements.video.cancelVideoFrameCallback(state.frameRequest);
  }
  if (state.animationRequest != null) cancelAnimationFrame(state.animationRequest);
  state.frameRequest = null;
  state.animationRequest = null;
  stopTracks();
  await releaseWakeLock();
  state.geometry = null;
  state.previousGeometry = null;
  state.frameQuality = null;
  overlayContext.clearRect(0, 0, elements.overlay.width, elements.overlay.height);
  elements.cameraStage.classList.remove("is-live");
  setControlsForStopped();
  setLiveState("idle", "Stopped");
  setCoach("Measurement stopped", "idle");
  if (!keepResult) resetSession();
  if (!keepModel && state.landmarker) {
    state.landmarker.close?.();
    state.landmarker = null;
    state.landmarkerPromise = null;
  }
}

function stopTracks() {
  const stream = state.stream || elements.video.srcObject;
  stream?.getTracks?.().forEach((track) => track.stop());
  elements.video.srcObject = null;
  state.stream = null;
}

function resetSession() {
  buffer.clear();
  state.currentEstimate = null;
  state.displayedBpm = null;
  state.rawEstimateHistory = [];
  state.sessionEstimates = [];
  state.motionScore = 1;
  state.lastSampleAt = -Infinity;
  clearAnalysisOnly();
  updateSessionSummary();
  elements.exportButton.disabled = true;
}

function clearAnalysisOnly() {
  state.currentEstimate = null;
  state.displayedBpm = null;
  state.rawEstimateHistory = [];
  elements.bpmValue.textContent = "—";
  elements.bpmDisplay.dataset.ready = "false";
  elements.confidenceValue.textContent = "Waiting";
  elements.confidenceBar.setAttribute("aria-valuenow", "0");
  elements.confidenceBar.querySelector("span").style.width = "0%";
  elements.collectionBar.style.width = "0%";
  elements.collectionTime.textContent = "0.0 s";
  elements.collectionLabel.textContent = "No signal collected";
  elements.peakLabel.textContent = "no peak";
  setQuality(elements.qualitySignal, "idle", "Waiting");
  clearCharts(elements.signalChart, elements.spectrumChart);
}

function updateSessionSummary() {
  if (state.sessionEstimates.length === 0) {
    elements.averageBpm.textContent = "—";
    elements.rangeBpm.textContent = "—";
    return;
  }
  const values = state.sessionEstimates.map((entry) => entry.bpm);
  elements.averageBpm.textContent = `${Math.round(average(values))} BPM`;
  elements.rangeBpm.textContent = `${Math.round(Math.min(...values))}–${Math.round(Math.max(...values))}`;
}

function getAnalysisOptions() {
  validateBpmRange();
  const windowSeconds = Number(elements.windowSeconds.value);
  return {
    minBpm: Number(elements.minBpm.value),
    maxBpm: Number(elements.maxBpm.value),
    windowSeconds,
    minimumSeconds: Math.min(8, windowSeconds - 2),
  };
}

function validateBpmRange() {
  let minimum = clamp(Number(elements.minBpm.value) || 45, 35, 180);
  let maximum = clamp(Number(elements.maxBpm.value) || 200, 60, 240);
  if (maximum < minimum + 20) maximum = minimum + 20;
  if (maximum > 240) {
    maximum = 240;
    minimum = Math.min(minimum, 220);
  }
  elements.minBpm.value = String(Math.round(minimum));
  elements.maxBpm.value = String(Math.round(maximum));
}

function exportCsv() {
  const csv = buffer.toCsv();
  if (!csv) return;
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const stamp = new Date().toISOString().replaceAll(":", "-").replace(/\.\d{3}Z$/, "Z");
  link.href = url;
  link.download = `pulsecam-session-${stamp}.csv`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function toggleFullscreen() {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await elements.cameraStage.requestFullscreen();
  } catch (error) {
    console.warn("Full screen was unavailable", error);
  }
}

async function requestWakeLock() {
  try {
    state.wakeLock = await navigator.wakeLock?.request("screen");
  } catch (error) {
    console.info("Screen wake lock was unavailable", error);
  }
}

async function releaseWakeLock() {
  try {
    await state.wakeLock?.release?.();
  } catch {
    // The browser may already have released it when the tab was hidden.
  }
  state.wakeLock = null;
}

async function handleVisibilityChange() {
  if (document.hidden) {
    buffer.clear();
    clearAnalysisOnly();
    await releaseWakeLock();
  } else if (state.running) {
    await requestWakeLock();
    setCoach("Signal reset after returning to the tab", "warn");
  }
}

function redrawCharts() {
  if (!state.currentEstimate) {
    clearCharts(elements.signalChart, elements.spectrumChart);
    return;
  }
  drawSignalChart(elements.signalChart, state.currentEstimate.waveform);
  drawSpectrumChart(elements.spectrumChart, state.currentEstimate.spectrum, state.displayedBpm || state.currentEstimate.bpm);
}

function setControlsForStarting() {
  elements.startButton.disabled = true;
  elements.stopButton.disabled = true;
  elements.cameraSelect.disabled = true;
}

function setControlsForStopped() {
  elements.startButton.disabled = false;
  elements.stopButton.disabled = true;
  elements.cameraSelect.disabled = elements.cameraSelect.options.length <= 1;
}

function setLiveState(status, label) {
  elements.liveState.dataset.state = status;
  elements.liveState.querySelector("b").textContent = label;
}

function setCoach(text, level = "idle") {
  if (elements.coachText.textContent !== text) elements.coachText.textContent = text;
  elements.cameraCoach.dataset.level = level;
}

function setQuality(element, level, label) {
  element.dataset.level = level;
  element.querySelector("dd").textContent = label;
}

function showFatalError(message) {
  setLiveState("error", "Error");
  setCoach(message, "bad");
}

function confidenceLabel(confidence, duration) {
  if (duration < 2) return "Waiting";
  if (confidence >= 0.78) return "High";
  if (confidence >= 0.6) return "Good";
  if (confidence >= DISPLAY_CONFIDENCE_THRESHOLD) return "Moderate";
  return "Low";
}

function signalQualityLabel(estimate) {
  if (estimate.duration < 3) return "Collecting";
  if (estimate.confidence >= 0.7) return "Strong";
  if (estimate.confidence >= 0.42) return "Usable";
  return "Weak";
}

function liveFpsLabel() {
  if (state.frameTimes.length < 2) return "— fps";
  const duration = (state.frameTimes.at(-1) - state.frameTimes[0]) / 1000;
  return duration > 0 ? `${((state.frameTimes.length - 1) / duration).toFixed(0)} fps` : "— fps";
}

function describeCameraError(error) {
  if (error?.name === "NotAllowedError") return "Camera permission was denied. Allow camera access in the browser, then try again.";
  if (error?.name === "NotFoundError") return "No usable camera was found on this device.";
  if (error?.name === "NotReadableError") return "The camera is already in use by another app or could not be opened.";
  if (error?.name === "OverconstrainedError") return "The selected camera cannot provide the requested video mode. Choose another camera.";
  return error?.message || "The camera could not be started.";
}

function rotatedRectangle(centre, ux, uy, halfWidth, halfHeight) {
  const horizontal = scalePoint(ux, halfWidth);
  const vertical = scalePoint(uy, halfHeight);
  return [
    subtract(subtract(centre, horizontal), vertical),
    add(subtract(centre, vertical), horizontal),
    add(add(centre, horizontal), vertical),
    add(subtract(centre, horizontal), vertical),
  ];
}

function isPointInPolygon(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i];
    const b = polygon[j];
    const intersects = ((a.y > y) !== (b.y > y)) && (x < (b.x - a.x) * (y - a.y) / ((b.y - a.y) || Number.EPSILON) + a.x);
    if (intersects) inside = !inside;
  }
  return inside;
}

function point(landmark) {
  return { x: landmark.x, y: landmark.y };
}

function add(a, b) {
  return { x: a.x + b.x, y: a.y + b.y };
}

function subtract(a, b) {
  return { x: a.x - b.x, y: a.y - b.y };
}

function scalePoint(p, scalar) {
  return { x: p.x * scalar, y: p.y * scalar };
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function magnitude(p) {
  return Math.hypot(p.x, p.y);
}

function dot(a, b) {
  return a.x * b.x + a.y * b.y;
}

function average(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}
