"""
Phase 5: FastAPI Backend Service for ECG Analysis
Wraps the DSP peak detector and HRV analyzer into a REST API endpoint.
Run: pip install fastapi uvicorn
"""

from fastapi import FastAPI, HTTPException
import numpy as np
from setup import load_ecg_record
from dsp import pan_tompkins_detector
from hrv_analyzer import compute_rr_intervals, compute_time_domain_hrv, compute_frequency_domain_hrv

app = FastAPI(
    title="ECG Analysis & HRV API",
    description="Backend API providing automated QRS detection, heartbeat classification, and HRV diagnostic metrics.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "ECG Backend Engine is running successfully."}

@app.get("/analyze/{record_name}")
def analyze_ecg_record(record_name: str):
    """
    Endpoint that loads a requested PhysioNet record, runs Pan-Tompkins QRS detection,
    computes time and frequency domain HRV metrics, and returns a JSON diagnostic report.
    """
    try:
        # 1. Load record data using Phase 1 helper
        ecg_signal, fs, true_annotations = load_ecg_record(record_name=record_name, db_dir='mitdb_cache')
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Record '{record_name}' could not be loaded: {str(e)}")
    
    # 2. Run Pan-Tompkins DSP pipeline from Phase 2
    filtered_signal, integrated_signal, computed_peaks = pan_tompkins_detector(ecg_signal, fs)
    
    # 3. Compute HRV metrics from Phase 4
    rr_intervals_ms = compute_rr_intervals(computed_peaks, fs)
    sdnn, rmssd = compute_time_domain_hrv(rr_intervals_ms)
    _, _, lf, hf, ratio = compute_frequency_domain_hrv(rr_intervals_ms)
    
    mean_hr = 60000 / np.mean(rr_intervals_ms) if len(rr_intervals_ms) > 0 else 0.0
    
    # 4. Construct JSON response payload
    report = {
        "record_name": record_name,
        "sampling_frequency_hz": fs,
        "total_beats_analyzed": len(computed_peaks),
        "mean_heart_rate_bpm": round(float(mean_hr), 2),
        "time_domain_metrics": {
            "sdnn_ms": round(float(sdnn), 2),
            "rmssd_ms": round(float(rmssd), 2)
        },
        "frequency_domain_metrics": {
            "lf_power_ms2": round(float(lf), 2),
            "hf_power_ms2": round(float(hf), 2),
            "lf_hf_ratio": round(float(ratio), 2)
        }
    }
    
    return report