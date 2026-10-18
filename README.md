# Cardiovascular Signal Analysis, AI Arrhythmia & HRV Dashboard

An end-to-end biomedical signal processing and deep learning pipeline for automated ECG QRS detection, Heart Rate Variability (HRV) autonomic profiling, and 1D-CNN arrhythmia classification using the PhysioNet MIT-BIH Arrhythmia Database.

## Key Architecture & Modules

* **`setup.py` (Local Persistence Layer):** Streams PhysioNet records directly and caches compressed NumPy binaries (`.npz`) locally to eliminate network latency and enable instant offline loading.
* **`dsp.py` (Pan-Tompkins DSP Engine):** Implements a 5-stage digital signal processing pipeline featuring zero-phase Butterworth bandpass filtering (5–15 Hz), derivative filtering, squaring, moving-window integration, and robust local R-wave crest snapping.
* **`hrv_analyzer.py` (Autonomic Diagnostics):** Computes RR-interval time-domain metrics (SDNN, RMSSD) and frequency-domain Power Spectral Density (PSD) using 4 Hz cubic spline interpolation to isolate LF, HF, and sympathovagal balance ratios.
* **`ml_classifier.py` (PyTorch 1D-CNN):** Extracts Z-normalized beat morphology windows around R-peaks to classify Normal Sinus beats versus Ectopic/PVC arrhythmias, achieving **99% accuracy and a 0.98 F1-score**.
* **`app_frontend.py` (Clinical Web UI):** Built with Streamlit for real-time telemetry inspection, AI confidence badging, interactive arrhythmia timestamp jumping, and one-click CSV/TXT clinical report exports.

## Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt