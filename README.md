# PulseCam

PulseCam is an experimental, browser-only heart-rate estimator. It tracks a face, samples colour from the forehead and both cheeks, and uses remote photoplethysmography (rPPG) to look for the small periodic colour change caused by blood flow.

**Live app:** [xosh.org/webcam-heart-rate](https://xosh.org/webcam-heart-rate/)

Video frames stay in the browser. There is no server-side processing, account, or upload.

> PulseCam is not a medical device. A webcam estimate can be wrong because of motion, lighting, camera auto-exposure, compression, frame timing, skin coverage, and many other factors. Do not use it for diagnosis or treatment decisions.

## What is different from older webcam pulse demos?

Many early demos used one fixed rectangle, the average green value, and the largest FFT bin. PulseCam adds several safeguards:

- MediaPipe's 478-point Face Landmarker follows the face instead of relying on a fixed crop or Haar cascade.
- Three independently sampled skin regions reduce the effect of a shadow, hair, glasses, or a local reflection.
- Real video-frame callbacks and timestamps are used; samples are resampled onto a uniform clock before frequency analysis.
- POS and CHROM colour projections suppress brightness changes that affect all colour channels together. A green-channel estimate is retained as a low-weight third opinion.
- Several regions and extraction methods vote on the spectral peak.
- Motion, head pose, lighting, clipping, frame timing, spectral dominance, and cross-method agreement contribute to an explicit confidence score.
- The app withholds the large BPM number until confidence is usable instead of presenting every FFT maximum as a measurement.
- The recovered waveform, frequency spectrum, quality checks, camera selection, configurable range/window, and raw-session CSV export are built in.

Eulerian video magnification is useful for *showing* tiny colour changes, but magnifying every pixel over time is not required to estimate pulse. This app instead extracts compact RGB time series from tracked skin and analyses those signals directly. That is substantially lighter in a browser and makes motion/quality rejection easier.

## Run locally

Camera access needs a secure context. `localhost` counts as secure, so any static server is enough:

```bash
npm start
```

Then open the local URL printed by `serve`.

There is no build step. `index.html`, `styles.css`, and the modules in `src/` are deployed directly by GitHub Pages.

The first load downloads the pinned MediaPipe Tasks Vision JavaScript/WASM package and Google's Face Landmarker model. After the browser has cached those files, repeat loads are normally much faster.

## Tests

The signal-processing tests generate irregularly timed RGB camera samples with a known pulse, illumination drift, harmonics, and noise:

```bash
npm test
```

They verify resampling, POS extraction, spectral peak recovery, end-to-end BPM estimation, and confidence penalties.

The same deterministic suite runs in GitHub Actions on every push and pull request.

## Signal pipeline

1. `requestVideoFrameCallback()` runs once per presented camera frame and supplies a high-resolution clock.
2. Face Landmarker runs at a throttled rate. The latest landmarks define a forehead region and two cheek regions in a face-relative coordinate system.
3. Each camera frame is downsampled once. Mean RGB, luma, valid-pixel coverage, and clipping are measured inside each polygon.
4. Irregular RGB samples are linearly resampled to a uniform 12–30 Hz series.
5. Overlapping 1.6-second windows produce POS and CHROM pulse signals. A moving-average high-pass removes slow illumination drift.
6. A Hann-windowed direct periodogram evaluates the configured BPM range at one-BPM spacing. POS, CHROM, green, and all three regions form a weighted ensemble.
7. Parabolic peak interpolation gives sub-bin resolution. Spectral SNR, competing peaks, method agreement, timing jitter, duration, light, motion, pose, and face size determine confidence.
8. Accepted estimates use a short robust temporal smoother for the display. Raw sampled RGB remains available through CSV export.

The processing code intentionally has no numerical-library dependency, which keeps the deployed page small and makes the estimator testable under Node.

## Project structure

```text
index.html          interface and accessible controls
styles.css          responsive visual design
src/app.js          camera, MediaPipe, ROI sampling, and UI state
src/rppg.js         resampling, POS/CHROM, spectra, confidence, CSV buffer
src/charts.js       dependency-free canvas plots
tests/              deterministic synthetic-signal tests
```

## Measurement advice

- Put the device on a stable surface.
- Use bright, steady light in front of the face. Avoid backlight and flickering lamps.
- Keep the full face visible, mostly forward, and roughly an arm's length away.
- Stay silent and still until at least one full analysis window has been collected.
- Compare the result with a validated sensor before trusting it.

## Research and prior work

The implementation was informed by:

- Wang, den Brinker, Stuijk, and de Haan, [*Algorithmic Principles of Remote PPG*](https://doi.org/10.1109/TBME.2016.2609282) — the POS method and optical model.
- de Haan and Jeanne, [*Robust Pulse Rate From Chrominance-Based rPPG*](https://doi.org/10.1109/TBME.2013.2266196) — the CHROM method.
- Google, [MediaPipe Face Landmarker for Web](https://developers.google.com/mediapipe/solutions/vision/face_landmarker/web_js).
- [p5WebcamPulse.js](https://github.com/ziyuan-linn/p5WebcamPulse.js), a working browser example with face and finger modes.
- [webcam-pulse-detector](https://github.com/thearn/webcam-pulse-detector), an influential forehead/FFT implementation.
- [webcam-heart-rate-monitor](https://github.com/giladoved/webcam-heart-rate-monitor), an Eulerian magnification demonstration.
- [RealTimeHeartRateMonitor](https://github.com/masonkadem/RealTimeHeartRateMonitor) and [eulerian-remote-heartrate-detection](https://github.com/rohintangirala/eulerian-remote-heartrate-detection), earlier OpenCV experiments.

PulseCam is a new implementation; code was not copied from those projects.

## Privacy

Camera frames are read only by JavaScript in the current page and are not transmitted by this project. Loading the page does contact GitHub Pages, jsDelivr (the MediaPipe runtime), and Google-hosted storage (the face model). Exported CSV files are created locally by the browser and contain timestamps, regional RGB averages, and quality scores—not images.

## Licence

MIT. See [LICENSE](LICENSE).
