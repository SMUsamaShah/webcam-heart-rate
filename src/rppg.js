const EPSILON = 1e-9;

export const DEFAULT_RPPG_OPTIONS = Object.freeze({
  minBpm: 45,
  maxBpm: 200,
  windowSeconds: 12,
  minimumSeconds: 7.5,
  spectrumStepBpm: 1,
});

export class RppgBuffer {
  constructor(maxDurationSeconds = 25) {
    this.maxDurationSeconds = maxDurationSeconds;
    this.samples = [];
  }

  add(timestampSeconds, rois, quality = {}) {
    if (!Number.isFinite(timestampSeconds) || !Array.isArray(rois) || rois.length === 0) return false;
    const previous = this.samples.at(-1);
    if (previous && timestampSeconds <= previous.t) return false;

    this.samples.push({
      t: timestampSeconds,
      rois: rois.map((roi) => ({
        r: Number(roi.r),
        g: Number(roi.g),
        b: Number(roi.b),
        luma: Number(roi.luma),
        validRatio: Number(roi.validRatio),
        clippedRatio: Number(roi.clippedRatio),
      })),
      quality: {
        faceScore: finiteOr(quality.faceScore, 1),
        lightScore: finiteOr(quality.lightScore, 1),
        motionScore: finiteOr(quality.motionScore, 1),
        poseScore: finiteOr(quality.poseScore, 1),
      },
    });

    const cutoff = timestampSeconds - this.maxDurationSeconds;
    let removeCount = 0;
    while (removeCount < this.samples.length && this.samples[removeCount].t < cutoff) removeCount += 1;
    if (removeCount > 0) this.samples.splice(0, removeCount);
    return true;
  }

  clear() {
    this.samples.length = 0;
  }

  get duration() {
    if (this.samples.length < 2) return 0;
    return this.samples.at(-1).t - this.samples[0].t;
  }

  recent(seconds) {
    if (this.samples.length === 0) return [];
    const cutoff = this.samples.at(-1).t - seconds;
    const start = this.samples.findIndex((sample) => sample.t >= cutoff);
    return this.samples.slice(Math.max(0, start));
  }

  toCsv() {
    if (this.samples.length === 0) return "";
    const roiCount = Math.max(...this.samples.map((sample) => sample.rois.length));
    const roiHeaders = [];
    for (let i = 0; i < roiCount; i += 1) {
      roiHeaders.push(`roi_${i + 1}_r`, `roi_${i + 1}_g`, `roi_${i + 1}_b`, `roi_${i + 1}_luma`, `roi_${i + 1}_valid_ratio`, `roi_${i + 1}_clipped_ratio`);
    }
    const headers = ["time_s", ...roiHeaders, "face_score", "light_score", "motion_score", "pose_score"];
    const t0 = this.samples[0].t;
    const rows = this.samples.map((sample) => {
      const roiValues = [];
      for (let i = 0; i < roiCount; i += 1) {
        const roi = sample.rois[i];
        if (!roi) roiValues.push("", "", "", "", "", "");
        else roiValues.push(roi.r, roi.g, roi.b, roi.luma, roi.validRatio, roi.clippedRatio);
      }
      return [
        (sample.t - t0).toFixed(5),
        ...roiValues,
        sample.quality.faceScore,
        sample.quality.lightScore,
        sample.quality.motionScore,
        sample.quality.poseScore,
      ].join(",");
    });
    return [headers.join(","), ...rows].join("\n");
  }
}

export function estimateHeartRate(inputSamples, userOptions = {}) {
  const options = normalizeOptions(userOptions);
  const samples = sanitiseSamples(inputSamples, options.windowSeconds);
  if (samples.length < 2) return emptyEstimate("Waiting for camera frames");

  const timing = analyseTiming(samples);
  const duration = timing.duration;
  if (duration < options.minimumSeconds) {
    return {
      ...emptyEstimate("Collecting a longer signal"),
      duration,
      fps: timing.fps,
      progress: clamp(duration / options.windowSeconds, 0, 1),
    };
  }

  if (timing.fps < 8) {
    return {
      ...emptyEstimate("Camera frame rate is too low"),
      duration,
      fps: timing.fps,
      progress: clamp(duration / options.windowSeconds, 0, 1),
    };
  }

  const targetFps = clamp(Math.round(timing.fps), 12, 30);
  const roiCount = Math.min(...samples.map((sample) => sample.rois.length));
  const grids = [];
  const methodPeaks = [];
  const waveforms = [];
  let bpmAxis = [];

  for (let roiIndex = 0; roiIndex < roiCount; roiIndex += 1) {
    const rgb = resampleRgb(samples, roiIndex, targetFps);
    if (!rgb || rgb.r.length < targetFps * options.minimumSeconds * 0.85) continue;

    const signals = [
      { name: `roi${roiIndex + 1}-pos`, weight: 0.52, data: extractProjectionSignal(rgb, targetFps, "pos") },
      { name: `roi${roiIndex + 1}-chrom`, weight: 0.33, data: extractProjectionSignal(rgb, targetFps, "chrom") },
      { name: `roi${roiIndex + 1}-green`, weight: 0.15, data: extractGreenSignal(rgb.g, targetFps) },
    ];

    waveforms.push(signals[0].data);
    for (const signal of signals) {
      const spectrum = computePowerSpectrum(signal.data, targetFps, options.minBpm, options.maxBpm, options.spectrumStepBpm);
      if (spectrum.power.length === 0) continue;
      bpmAxis = spectrum.bpm;
      const normalised = normaliseSpectrum(spectrum.power);
      const peakIndex = indexOfMax(normalised);
      grids.push({ power: normalised, weight: signal.weight, name: signal.name });
      methodPeaks.push({ name: signal.name, bpm: bpmAxis[peakIndex], weight: signal.weight });
    }
  }

  if (grids.length === 0 || bpmAxis.length < 3) {
    return {
      ...emptyEstimate("Pulse signal could not be separated"),
      duration,
      fps: timing.fps,
      progress: clamp(duration / options.windowSeconds, 0, 1),
    };
  }

  const combined = weightedSpectrumMean(grids);
  const candidate = scoreCandidates(combined, bpmAxis, options.previousBpm);
  const peakIndex = indexOfMax(candidate);
  const bpm = parabolicPeak(bpmAxis, candidate, peakIndex);
  const peakPower = combined[peakIndex];
  const noiseFloor = median(combined.filter((_, index) => Math.abs(index - peakIndex) > 4));
  const snrDb = 10 * Math.log10((peakPower + EPSILON) / (noiseFloor + EPSILON));
  const secondPeak = findSecondPeak(combined, peakIndex, Math.ceil(9 / options.spectrumStepBpm));
  const dominance = peakPower / (secondPeak + EPSILON);
  const consensusError = weightedConsensusError(methodPeaks, bpm);
  const sampleQuality = averageSampleQuality(samples);
  const durationScore = clamp((duration - options.minimumSeconds + 1.5) / 5, 0, 1);
  const timingScore = clamp((timing.fps - 8) / 14, 0, 1) * clamp(1 - timing.jitter / 0.45, 0, 1);
  const spectralScore = clamp((snrDb - 1.5) / 10.5, 0, 1);
  const dominanceScore = clamp((dominance - 1) / 2.5, 0, 1);
  const consensusScore = clamp(1 - consensusError / 20, 0, 1);
  const stillnessScore = Math.sqrt(sampleQuality.motionScore * sampleQuality.poseScore);

  let confidence =
    0.34 * spectralScore +
    0.14 * dominanceScore +
    0.17 * consensusScore +
    0.1 * timingScore +
    0.1 * sampleQuality.lightScore +
    0.15 * stillnessScore;
  confidence *= 0.48 + 0.52 * durationScore;
  confidence *= 0.45 + 0.55 * sampleQuality.faceScore;

  if (Number.isFinite(options.previousBpm)) {
    const jump = Math.abs(bpm - options.previousBpm);
    if (jump > 28) confidence *= clamp(1 - (jump - 28) / 70, 0.72, 1);
  }
  confidence = clamp(confidence, 0, 1);

  const reasons = [];
  if (sampleQuality.motionScore < 0.62) reasons.push("Hold your head and camera still");
  if (sampleQuality.poseScore < 0.6) reasons.push("Look more directly at the camera");
  if (sampleQuality.lightScore < 0.6) reasons.push("Use brighter, steadier front lighting");
  if (timing.fps < 14) reasons.push("The camera frame rate is low");
  if (spectralScore < 0.4) reasons.push("No clear repeating pulse peak yet");
  if (consensusScore < 0.4) reasons.push("Skin regions do not agree yet");

  return {
    bpm: Number.isFinite(bpm) ? bpm : null,
    confidence,
    duration,
    fps: timing.fps,
    progress: clamp(duration / options.windowSeconds, 0, 1),
    waveform: combineWaveforms(waveforms, Math.round(targetFps * 8)),
    spectrum: { bpm: bpmAxis, power: scaleToUnit(combined) },
    diagnostics: {
      snrDb,
      dominance,
      consensusError,
      methodPeaks,
      timingJitter: timing.jitter,
      quality: sampleQuality,
      reasons,
    },
    status: confidence >= 0.62 ? "Strong pulse signal" : confidence >= 0.4 ? "Estimate stabilising" : reasons[0] || "Keep still while the signal builds",
  };
}

export function extractProjectionSignal(rgb, fps, method = "pos") {
  const length = Math.min(rgb.r.length, rgb.g.length, rgb.b.length);
  if (length === 0) return [];
  const windowLength = clamp(Math.round(fps * 1.6), 12, length);
  const step = Math.max(1, Math.round(windowLength / 4));
  const output = new Float64Array(length);
  const weights = new Float64Array(length);
  const starts = [];
  for (let start = 0; start <= length - windowLength; start += step) starts.push(start);
  if (starts.at(-1) !== length - windowLength) starts.push(length - windowLength);

  for (const start of starts) {
    const end = start + windowLength;
    const meanR = meanRange(rgb.r, start, end) || EPSILON;
    const meanG = meanRange(rgb.g, start, end) || EPSILON;
    const meanB = meanRange(rgb.b, start, end) || EPSILON;
    const x = new Float64Array(windowLength);
    const y = new Float64Array(windowLength);

    for (let i = 0; i < windowLength; i += 1) {
      const r = rgb.r[start + i] / meanR - 1;
      const g = rgb.g[start + i] / meanG - 1;
      const b = rgb.b[start + i] / meanB - 1;
      if (method === "chrom") {
        x[i] = 3 * r - 2 * g;
        y[i] = 1.5 * r + g - 1.5 * b;
      } else {
        x[i] = g - b;
        y[i] = g + b - 2 * r;
      }
    }

    const alpha = standardDeviation(x) / (standardDeviation(y) + EPSILON);
    const segmentMean = mean(x.map((value, index) => method === "chrom" ? value - alpha * y[index] : value + alpha * y[index]));
    for (let i = 0; i < windowLength; i += 1) {
      const taper = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / Math.max(1, windowLength - 1));
      const projected = method === "chrom" ? x[i] - alpha * y[i] : x[i] + alpha * y[i];
      output[start + i] += (projected - segmentMean) * taper;
      weights[start + i] += taper;
    }
  }

  const projected = Array.from(output, (value, index) => value / (weights[index] || 1));
  return standardise(highPassMovingAverage(projected, Math.max(3, Math.round(fps * 1.8))));
}

export function computePowerSpectrum(signal, fps, minBpm = 45, maxBpm = 200, stepBpm = 1) {
  const clean = linearDetrend(signal.map((value) => finiteOr(value, 0)));
  const length = clean.length;
  if (length < 4 || !Number.isFinite(fps) || fps <= 0) return { bpm: [], power: [] };

  const windowed = clean.map((value, index) => {
    const hann = 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / Math.max(1, length - 1));
    return value * hann;
  });
  const bpm = [];
  const power = [];
  for (let candidate = minBpm; candidate <= maxBpm + stepBpm * 0.25; candidate += stepBpm) {
    const frequency = candidate / 60;
    let real = 0;
    let imaginary = 0;
    for (let i = 0; i < length; i += 1) {
      const angle = (2 * Math.PI * frequency * i) / fps;
      real += windowed[i] * Math.cos(angle);
      imaginary -= windowed[i] * Math.sin(angle);
    }
    bpm.push(candidate);
    power.push((real * real + imaginary * imaginary) / (length * length));
  }
  return { bpm, power: smoothThreePoint(power) };
}

export function resampleSeries(times, values, fps, start = times[0], end = times.at(-1)) {
  if (times.length !== values.length || times.length < 2 || fps <= 0 || end <= start) return [];
  const count = Math.max(2, Math.floor((end - start) * fps) + 1);
  const result = new Array(count);
  let sourceIndex = 0;
  for (let i = 0; i < count; i += 1) {
    const t = Math.min(end, start + i / fps);
    while (sourceIndex < times.length - 2 && times[sourceIndex + 1] < t) sourceIndex += 1;
    const t0 = times[sourceIndex];
    const t1 = times[sourceIndex + 1];
    const ratio = t1 === t0 ? 0 : clamp((t - t0) / (t1 - t0), 0, 1);
    result[i] = values[sourceIndex] + ratio * (values[sourceIndex + 1] - values[sourceIndex]);
  }
  return result;
}

function extractGreenSignal(green, fps) {
  const meanGreen = mean(green) || EPSILON;
  const normalised = green.map((value) => value / meanGreen - 1);
  return standardise(highPassMovingAverage(normalised, Math.max(3, Math.round(fps * 1.8))));
}

function sanitiseSamples(inputSamples, windowSeconds) {
  if (!Array.isArray(inputSamples)) return [];
  const valid = inputSamples.filter((sample) =>
    sample && Number.isFinite(sample.t) && Array.isArray(sample.rois) && sample.rois.length > 0 &&
    sample.rois.every((roi) => Number.isFinite(roi.r) && Number.isFinite(roi.g) && Number.isFinite(roi.b))
  ).sort((a, b) => a.t - b.t);
  if (valid.length === 0) return [];
  const cutoff = valid.at(-1).t - windowSeconds;
  return valid.filter((sample) => sample.t >= cutoff);
}

function analyseTiming(samples) {
  const intervals = [];
  for (let i = 1; i < samples.length; i += 1) {
    const delta = samples[i].t - samples[i - 1].t;
    if (delta > 0 && delta < 1) intervals.push(delta);
  }
  const duration = samples.at(-1).t - samples[0].t;
  const fps = duration > 0 ? (samples.length - 1) / duration : 0;
  const averageInterval = mean(intervals);
  const jitter = averageInterval > 0 ? standardDeviation(intervals) / averageInterval : 1;
  return { duration, fps, jitter };
}

function resampleRgb(samples, roiIndex, fps) {
  const selected = samples.filter((sample) => {
    const roi = sample.rois[roiIndex];
    return roi && Number.isFinite(roi.r) && Number.isFinite(roi.g) && Number.isFinite(roi.b);
  });
  if (selected.length < 2) return null;
  const times = selected.map((sample) => sample.t);
  const start = times[0];
  const end = times.at(-1);
  return {
    r: resampleSeries(times, selected.map((sample) => sample.rois[roiIndex].r), fps, start, end),
    g: resampleSeries(times, selected.map((sample) => sample.rois[roiIndex].g), fps, start, end),
    b: resampleSeries(times, selected.map((sample) => sample.rois[roiIndex].b), fps, start, end),
  };
}

function scoreCandidates(spectrum, bpmAxis, previousBpm) {
  return spectrum.map((power, index) => {
    const bpm = bpmAxis[index];
    const harmonicIndex = nearestIndex(bpmAxis, bpm * 2);
    const harmonicSupport = bpm * 2 <= bpmAxis.at(-1) ? spectrum[harmonicIndex] * 0.16 : 0;
    const prior = Number.isFinite(previousBpm) ? 0.84 + 0.16 * Math.exp(-0.5 * ((bpm - previousBpm) / 16) ** 2) : 1;
    return (power + harmonicSupport) * prior;
  });
}

function weightedSpectrumMean(grids) {
  const length = Math.min(...grids.map((grid) => grid.power.length));
  const result = new Array(length).fill(0);
  let totalWeight = 0;
  for (const grid of grids) {
    totalWeight += grid.weight;
    for (let i = 0; i < length; i += 1) result[i] += grid.power[i] * grid.weight;
  }
  return result.map((value) => value / (totalWeight || 1));
}

function normaliseSpectrum(power) {
  const floor = median(power);
  const lifted = power.map((value) => Math.max(0, value - floor * 0.6));
  const total = lifted.reduce((sum, value) => sum + value, 0) || 1;
  return lifted.map((value) => value / total);
}

function weightedConsensusError(peaks, referenceBpm) {
  let weightedError = 0;
  let totalWeight = 0;
  for (const peak of peaks) {
    const direct = Math.abs(peak.bpm - referenceBpm);
    const half = Math.abs(peak.bpm / 2 - referenceBpm);
    const double = Math.abs(peak.bpm * 2 - referenceBpm);
    weightedError += Math.min(direct, half, double) * peak.weight;
    totalWeight += peak.weight;
  }
  return weightedError / (totalWeight || 1);
}

function averageSampleQuality(samples) {
  const defaults = { faceScore: 1, lightScore: 1, motionScore: 1, poseScore: 1 };
  const result = {};
  for (const key of Object.keys(defaults)) {
    result[key] = clamp(mean(samples.map((sample) => finiteOr(sample.quality?.[key], defaults[key]))), 0, 1);
  }
  return result;
}

function combineWaveforms(waveforms, maxLength) {
  if (waveforms.length === 0) return [];
  const length = Math.min(maxLength, ...waveforms.map((waveform) => waveform.length));
  const result = new Array(length).fill(0);
  for (const waveform of waveforms) {
    const offset = waveform.length - length;
    for (let i = 0; i < length; i += 1) result[i] += waveform[offset + i] / waveforms.length;
  }
  return standardise(result);
}

function highPassMovingAverage(signal, width) {
  if (signal.length === 0) return [];
  const prefix = new Float64Array(signal.length + 1);
  for (let i = 0; i < signal.length; i += 1) prefix[i + 1] = prefix[i] + signal[i];
  const half = Math.max(1, Math.floor(width / 2));
  return signal.map((value, index) => {
    const start = Math.max(0, index - half);
    const end = Math.min(signal.length, index + half + 1);
    return value - (prefix[end] - prefix[start]) / (end - start);
  });
}

function linearDetrend(signal) {
  const length = signal.length;
  if (length < 2) return signal.slice();
  const meanX = (length - 1) / 2;
  const meanY = mean(signal);
  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < length; i += 1) {
    numerator += (i - meanX) * (signal[i] - meanY);
    denominator += (i - meanX) ** 2;
  }
  const slope = numerator / (denominator || 1);
  return signal.map((value, index) => value - (meanY + slope * (index - meanX)));
}

function standardise(signal) {
  const average = mean(signal);
  const deviation = standardDeviation(signal) || 1;
  return signal.map((value) => (value - average) / deviation);
}

function scaleToUnit(values) {
  const maximum = Math.max(...values, EPSILON);
  return values.map((value) => value / maximum);
}

function smoothThreePoint(values) {
  if (values.length < 3) return values.slice();
  return values.map((value, index) => {
    if (index === 0 || index === values.length - 1) return value;
    return values[index - 1] * 0.2 + value * 0.6 + values[index + 1] * 0.2;
  });
}

function parabolicPeak(axis, values, index) {
  if (index <= 0 || index >= values.length - 1) return axis[index];
  const left = values[index - 1];
  const centre = values[index];
  const right = values[index + 1];
  const denominator = left - 2 * centre + right;
  if (Math.abs(denominator) < EPSILON) return axis[index];
  const offset = clamp(0.5 * (left - right) / denominator, -1, 1);
  return axis[index] + offset * (axis[index + 1] - axis[index]);
}

function findSecondPeak(values, peakIndex, exclusionRadius) {
  let second = 0;
  for (let i = 0; i < values.length; i += 1) {
    if (Math.abs(i - peakIndex) > exclusionRadius) second = Math.max(second, values[i]);
  }
  return second;
}

function nearestIndex(values, target) {
  if (values.length < 2) return 0;
  const step = values[1] - values[0];
  return clamp(Math.round((target - values[0]) / step), 0, values.length - 1);
}

function indexOfMax(values) {
  let bestIndex = 0;
  for (let i = 1; i < values.length; i += 1) if (values[i] > values[bestIndex]) bestIndex = i;
  return bestIndex;
}

function normalizeOptions(userOptions) {
  const options = { ...DEFAULT_RPPG_OPTIONS, ...userOptions };
  options.minBpm = clamp(Number(options.minBpm), 35, 180);
  options.maxBpm = clamp(Number(options.maxBpm), options.minBpm + 20, 240);
  options.windowSeconds = clamp(Number(options.windowSeconds), 8, 30);
  options.minimumSeconds = Math.min(clamp(Number(options.minimumSeconds), 6, 15), options.windowSeconds - 0.5);
  options.spectrumStepBpm = clamp(Number(options.spectrumStepBpm), 0.25, 2);
  options.previousBpm = Number(options.previousBpm);
  return options;
}

function emptyEstimate(status) {
  return {
    bpm: null,
    confidence: 0,
    duration: 0,
    fps: 0,
    progress: 0,
    waveform: [],
    spectrum: { bpm: [], power: [] },
    diagnostics: { reasons: [] },
    status,
  };
}

function mean(values) {
  if (!values || values.length === 0) return 0;
  let total = 0;
  for (const value of values) total += value;
  return total / values.length;
}

function meanRange(values, start, end) {
  let total = 0;
  for (let i = start; i < end; i += 1) total += values[i];
  return total / Math.max(1, end - start);
}

function standardDeviation(values) {
  if (!values || values.length < 2) return 0;
  const average = mean(values);
  let sumSquares = 0;
  for (const value of values) sumSquares += (value - average) ** 2;
  return Math.sqrt(sumSquares / values.length);
}

function median(values) {
  if (!values || values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function finiteOr(value, fallback) {
  return Number.isFinite(value) ? value : fallback;
}
