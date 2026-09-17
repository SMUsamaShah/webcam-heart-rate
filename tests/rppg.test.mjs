import test from "node:test";
import assert from "node:assert/strict";

import {
  computePowerSpectrum,
  estimateHeartRate,
  extractProjectionSignal,
  resampleSeries,
} from "../src/rppg.js";

test("resampleSeries interpolates irregular samples onto a uniform clock", () => {
  const result = resampleSeries([0, 0.4, 1], [0, 4, 10], 2, 0, 1);
  assert.deepEqual(result.map((value) => Number(value.toFixed(6))), [0, 5, 10]);
});

test("power spectrum locates a synthetic pulse frequency", () => {
  const fps = 30;
  const bpm = 78;
  const signal = Array.from({ length: fps * 12 }, (_, index) => Math.sin(2 * Math.PI * (bpm / 60) * index / fps));
  const spectrum = computePowerSpectrum(signal, fps, 45, 180, 1);
  const peakIndex = spectrum.power.indexOf(Math.max(...spectrum.power));
  assert.ok(Math.abs(spectrum.bpm[peakIndex] - bpm) <= 1, `peak was ${spectrum.bpm[peakIndex]} BPM`);
});

test("POS projection and ensemble recover heart rate from noisy RGB samples", () => {
  const expectedBpm = 72;
  const samples = syntheticCameraSamples({ bpm: expectedBpm, duration: 14, fps: 30, noise: 0.32 });
  const estimate = estimateHeartRate(samples, {
    minBpm: 45,
    maxBpm: 180,
    windowSeconds: 12,
    minimumSeconds: 7,
  });
  assert.ok(Number.isFinite(estimate.bpm));
  assert.ok(Math.abs(estimate.bpm - expectedBpm) < 3, `estimated ${estimate.bpm} BPM`);
  assert.ok(estimate.confidence > 0.45, `confidence was ${estimate.confidence}`);
  assert.ok(estimate.waveform.length > 100);
});

test("quality penalties reduce confidence without moving the spectral peak", () => {
  const clean = syntheticCameraSamples({ bpm: 96, duration: 14, fps: 30, noise: 0.22 });
  const poor = clean.map((sample) => ({
    ...sample,
    quality: { ...sample.quality, motionScore: 0.08, poseScore: 0.3, lightScore: 0.35 },
  }));
  const cleanEstimate = estimateHeartRate(clean, { windowSeconds: 12, minimumSeconds: 7 });
  const poorEstimate = estimateHeartRate(poor, { windowSeconds: 12, minimumSeconds: 7 });
  assert.ok(Math.abs(poorEstimate.bpm - 96) < 3);
  assert.ok(poorEstimate.confidence < cleanEstimate.confidence);
});

test("projection signal preserves the injected periodic component", () => {
  const fps = 25;
  const bpm = 84;
  const length = fps * 10;
  const rgb = { r: [], g: [], b: [] };
  for (let index = 0; index < length; index += 1) {
    const pulse = Math.sin(2 * Math.PI * (bpm / 60) * index / fps);
    const drift = Math.sin(2 * Math.PI * 0.12 * index / fps) * 5;
    rgb.r.push(145 + drift + pulse * 0.7);
    rgb.g.push(112 + drift + pulse * 1.7);
    rgb.b.push(82 + drift + pulse * 0.35);
  }
  const projected = extractProjectionSignal(rgb, fps, "pos");
  const spectrum = computePowerSpectrum(projected, fps, 45, 180, 1);
  const peak = spectrum.bpm[spectrum.power.indexOf(Math.max(...spectrum.power))];
  assert.ok(Math.abs(peak - bpm) <= 2, `peak was ${peak} BPM`);
});

function syntheticCameraSamples({ bpm, duration, fps, noise }) {
  let seed = 0x12345678;
  const random = () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return ((seed >>> 0) / 0xffffffff) * 2 - 1;
  };

  const samples = [];
  let time = 0;
  const count = Math.floor(duration * fps);
  for (let index = 0; index < count; index += 1) {
    const jitter = random() * 0.0025;
    time += 1 / fps + jitter;
    const pulse = Math.sin(2 * Math.PI * (bpm / 60) * time);
    const harmonic = 0.22 * Math.sin(2 * Math.PI * (bpm / 30) * time + 0.4);
    const lighting = 4.5 * Math.sin(2 * Math.PI * 0.13 * time) + 1.8 * Math.sin(2 * Math.PI * 0.27 * time);
    const rois = [0, 1, 2].map((roiIndex) => {
      const phasePulse = pulse + harmonic + 0.04 * Math.sin(2 * Math.PI * (bpm / 60) * time + roiIndex * 0.15);
      return {
        r: 148 + lighting + phasePulse * 0.75 + random() * noise,
        g: 111 + lighting + phasePulse * 1.85 + random() * noise,
        b: 83 + lighting + phasePulse * 0.38 + random() * noise,
        luma: 113,
        validRatio: 0.97,
        clippedRatio: 0,
      };
    });
    samples.push({
      t: time,
      rois,
      quality: { faceScore: 0.96, lightScore: 0.95, motionScore: 0.94, poseScore: 0.95 },
    });
  }
  return samples;
}
