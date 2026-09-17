const COLOURS = Object.freeze({
  grid: "rgba(183, 230, 212, 0.09)",
  axis: "rgba(183, 230, 212, 0.32)",
  text: "rgba(196, 214, 207, 0.62)",
  mint: "#62e9aa",
  mintFill: "rgba(85, 230, 165, 0.12)",
  peak: "rgba(122, 244, 186, 0.7)",
  empty: "rgba(148, 170, 161, 0.35)",
});

export function drawSignalChart(canvas, signal = []) {
  const { context, width, height } = prepareCanvas(canvas);
  const bounds = { left: 34, right: width - 10, top: 12, bottom: height - 24 };
  drawGrid(context, bounds, 4, 3);
  drawTimeLabels(context, bounds);

  if (!signal || signal.length < 3) {
    drawEmpty(context, width, height, "Signal appears after a few seconds");
    return;
  }

  const clipped = robustClip(signal, 2.8);
  const gradient = context.createLinearGradient(0, bounds.top, 0, bounds.bottom);
  gradient.addColorStop(0, "rgba(85, 230, 165, 0.22)");
  gradient.addColorStop(1, "rgba(85, 230, 165, 0)");

  context.beginPath();
  clipped.forEach((value, index) => {
    const x = bounds.left + (index / Math.max(1, clipped.length - 1)) * (bounds.right - bounds.left);
    const y = bounds.top + (0.5 - value / 6) * (bounds.bottom - bounds.top);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.lineTo(bounds.right, bounds.bottom);
  context.lineTo(bounds.left, bounds.bottom);
  context.closePath();
  context.fillStyle = gradient;
  context.fill();

  context.beginPath();
  clipped.forEach((value, index) => {
    const x = bounds.left + (index / Math.max(1, clipped.length - 1)) * (bounds.right - bounds.left);
    const y = bounds.top + (0.5 - value / 6) * (bounds.bottom - bounds.top);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.strokeStyle = COLOURS.mint;
  context.lineWidth = 1.6;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.stroke();
}

export function drawSpectrumChart(canvas, spectrum = { bpm: [], power: [] }, peakBpm = null) {
  const { context, width, height } = prepareCanvas(canvas);
  const bounds = { left: 34, right: width - 10, top: 12, bottom: height - 24 };
  drawGrid(context, bounds, 4, 3);

  const bpm = spectrum?.bpm || [];
  const power = spectrum?.power || [];
  if (bpm.length < 2 || power.length !== bpm.length) {
    drawEmpty(context, width, height, "Frequency peak appears with the estimate");
    return;
  }

  const minBpm = bpm[0];
  const maxBpm = bpm.at(-1);
  drawBpmLabels(context, bounds, minBpm, maxBpm);

  const gradient = context.createLinearGradient(0, bounds.top, 0, bounds.bottom);
  gradient.addColorStop(0, "rgba(85, 230, 165, 0.26)");
  gradient.addColorStop(1, "rgba(85, 230, 165, 0.015)");

  context.beginPath();
  power.forEach((value, index) => {
    const x = bounds.left + (index / (power.length - 1)) * (bounds.right - bounds.left);
    const y = bounds.bottom - Math.max(0, Math.min(1, value)) * (bounds.bottom - bounds.top);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.lineTo(bounds.right, bounds.bottom);
  context.lineTo(bounds.left, bounds.bottom);
  context.closePath();
  context.fillStyle = gradient;
  context.fill();

  context.beginPath();
  power.forEach((value, index) => {
    const x = bounds.left + (index / (power.length - 1)) * (bounds.right - bounds.left);
    const y = bounds.bottom - Math.max(0, Math.min(1, value)) * (bounds.bottom - bounds.top);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.strokeStyle = COLOURS.mint;
  context.lineWidth = 1.5;
  context.stroke();

  if (Number.isFinite(peakBpm) && peakBpm >= minBpm && peakBpm <= maxBpm) {
    const x = bounds.left + ((peakBpm - minBpm) / (maxBpm - minBpm)) * (bounds.right - bounds.left);
    context.setLineDash([3, 4]);
    context.beginPath();
    context.moveTo(x, bounds.top);
    context.lineTo(x, bounds.bottom);
    context.strokeStyle = COLOURS.peak;
    context.lineWidth = 1;
    context.stroke();
    context.setLineDash([]);
  }
}

export function clearCharts(signalCanvas, spectrumCanvas) {
  drawSignalChart(signalCanvas, []);
  drawSpectrumChart(spectrumCanvas, { bpm: [], power: [] });
}

function prepareCanvas(canvas) {
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const cssWidth = Math.max(260, Math.round(canvas.clientWidth || 600));
  const cssHeight = Math.max(120, Math.round(canvas.clientHeight || 172));
  const pixelWidth = Math.round(cssWidth * ratio);
  const pixelHeight = Math.round(cssHeight * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, cssWidth, cssHeight);
  return { context, width: cssWidth, height: cssHeight };
}

function drawGrid(context, bounds, verticalDivisions, horizontalDivisions) {
  context.save();
  context.strokeStyle = COLOURS.grid;
  context.lineWidth = 1;
  for (let i = 0; i <= verticalDivisions; i += 1) {
    const x = bounds.left + (i / verticalDivisions) * (bounds.right - bounds.left);
    context.beginPath();
    context.moveTo(Math.round(x) + 0.5, bounds.top);
    context.lineTo(Math.round(x) + 0.5, bounds.bottom);
    context.stroke();
  }
  for (let i = 0; i <= horizontalDivisions; i += 1) {
    const y = bounds.top + (i / horizontalDivisions) * (bounds.bottom - bounds.top);
    context.beginPath();
    context.moveTo(bounds.left, Math.round(y) + 0.5);
    context.lineTo(bounds.right, Math.round(y) + 0.5);
    context.stroke();
  }
  context.restore();
}

function drawTimeLabels(context, bounds) {
  context.fillStyle = COLOURS.text;
  context.font = "10px ui-sans-serif, system-ui, sans-serif";
  context.textBaseline = "top";
  context.fillText("−8 s", bounds.left, bounds.bottom + 7);
  context.textAlign = "right";
  context.fillText("now", bounds.right, bounds.bottom + 7);
  context.textAlign = "left";
}

function drawBpmLabels(context, bounds, minimum, maximum) {
  context.fillStyle = COLOURS.text;
  context.font = "10px ui-sans-serif, system-ui, sans-serif";
  context.textBaseline = "top";
  for (let i = 0; i <= 4; i += 1) {
    const ratio = i / 4;
    const value = Math.round(minimum + ratio * (maximum - minimum));
    const x = bounds.left + ratio * (bounds.right - bounds.left);
    context.textAlign = i === 0 ? "left" : i === 4 ? "right" : "center";
    context.fillText(String(value), x, bounds.bottom + 7);
  }
  context.textAlign = "left";
}

function drawEmpty(context, width, height, label) {
  context.fillStyle = COLOURS.empty;
  context.font = "11px ui-sans-serif, system-ui, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(label, width / 2, height / 2);
  context.textAlign = "left";
}

function robustClip(signal, limit) {
  const sorted = [...signal].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)] || 0;
  const deviations = sorted.map((value) => Math.abs(value - median)).sort((a, b) => a - b);
  const mad = deviations[Math.floor(deviations.length / 2)] || 1;
  const scale = Math.max(0.25, 1.4826 * mad);
  return signal.map((value) => Math.max(-limit, Math.min(limit, (value - median) / scale)));
}
