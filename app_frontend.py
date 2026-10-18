"""
Streamlit Clinical Dashboard: Pan-Tompkins DSP + PyTorch 1D-CNN + HRV Suite
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from setup import load_ecg_record, MITBIH_RECORDS, get_cached_records
from dsp import pan_tompkins_detector
from hrv_analyzer import compute_rr_intervals, compute_time_domain_hrv, compute_frequency_domain_hrv
from ml_classifier import predict_arrhythmias

st.set_page_config(page_title="ECG AI & HRV Clinical Dashboard", layout="wide")

st.title("Cardiovascular Signal Analysis, AI Arrhythmia & HRV Dashboard")
st.markdown("End-to-end clinical pipeline powered by **Pan-Tompkins DSP**, **PyTorch 1D-CNN**, and **HRV Spectral Analysis**.")

st.sidebar.header("Patient Record & Controls")
cached_list = get_cached_records('mitdb_cache')

# Prioritize already cached records at the top of the dropdown so nothing hangs
sorted_records = cached_list + [r for r in MITBIH_RECORDS if r not in cached_list]

def format_record_label(rec):
    return f"Record {rec} (✅ Instant Local)" if rec in cached_list else f"Record {rec} (☁️ Stream from Cloud)"

record_choice = st.sidebar.selectbox(
    "Select MIT-BIH Patient Record",
    options=sorted_records,
    format_func=format_record_label
)

start_sec = st.sidebar.slider("Start Time Offset (Seconds)", min_value=0, max_value=600, value=0, step=5)
duration = st.sidebar.slider("Window Duration (Seconds)", min_value=3, max_value=15, value=6)

@st.cache_data(show_spinner=False)
def run_full_clinical_pipeline(rec):
    signal, fs, annotations = load_ecg_record(record_name=rec, db_dir='mitdb_cache')
    filtered, integrated, raw_peaks = pan_tompkins_detector(signal, fs)
    
    # Guarantee exact R-wave crest alignment (+/- 80ms local peak snap)
    radius = int(0.08 * fs)
    aligned_peaks = np.array([
        max(0, p - radius) + np.argmax(signal[max(0, p - radius):min(len(signal), p + radius)])
        for p in raw_peaks
    ])
    
    # Run HRV Engine
    rr = compute_rr_intervals(aligned_peaks, fs)
    sdnn, rmssd = compute_time_domain_hrv(rr)
    freqs, psd, lf, hf, ratio = compute_frequency_domain_hrv(rr)
    
    # Run PyTorch 1D-CNN Arrhythmia Classifier
    valid_peaks, beat_classes = predict_arrhythmias(signal, aligned_peaks, fs)
    
    return signal, fs, aligned_peaks, rr, sdnn, rmssd, freqs, psd, lf, hf, ratio, valid_peaks, beat_classes

with st.spinner(f"Running DSP, 1D-CNN Inference & HRV Analysis on Record {record_choice}..."):
    signal, fs, peaks, rr, sdnn, rmssd, freqs, psd, lf, hf, ratio, valid_peaks, beat_classes = run_full_clinical_pipeline(record_choice)

num_abnormal = int(np.sum(beat_classes == 1))
burden_pct = (num_abnormal / len(beat_classes) * 100) if len(beat_classes) > 0 else 0.0

# Top KPI Row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mean Heart Rate", f"{60000 / np.mean(rr):.1f} BPM")
c2.metric("Total Detected Beats", f"{len(peaks)}")
c3.metric("AI Detected Arrhythmias", f"{num_abnormal} Beats", f"{burden_pct:.1f}% burden", delta_color="inverse")
c4.metric("SDNN (Time HRV)", f"{sdnn:.2f} ms")
c5.metric("RMSSD (Vagal Tone)", f"{rmssd:.2f} ms")

# 1. Main ECG Waveform & AI Beat Classification Plot
st.subheader(f"ECG Telemetry & PyTorch 1D-CNN Beat Classification ({start_sec}s – {start_sec + duration}s)")
s_idx = int(start_sec * fs)
e_idx = int((start_sec + duration) * fs)
time_axis = np.arange(s_idx, e_idx) / fs

fig_ecg, ax_ecg = plt.subplots(figsize=(12, 3.5))
ax_ecg.plot(time_axis, signal[s_idx:e_idx], color='#1f77b4', linewidth=1.2, label='Raw ECG (Lead MLII)')

mask = (valid_peaks >= s_idx) & (valid_peaks < e_idx)
win_peaks = valid_peaks[mask]
win_classes = beat_classes[mask]

norm_peaks = win_peaks[win_classes == 0]
abn_peaks = win_peaks[win_classes == 1]

ax_ecg.scatter(norm_peaks / fs, signal[norm_peaks], color='#2ca02c', s=55, zorder=5, label='Normal Beat (1D-CNN)')
if len(abn_peaks) > 0:
    ax_ecg.scatter(abn_peaks / fs, signal[abn_peaks], color='#d62728', marker='^', s=90, zorder=6, label='Arrhythmia / PVC (1D-CNN)')

ax_ecg.set_xlabel("Time (seconds)")
ax_ecg.set_ylabel("Amplitude (mV)")
ax_ecg.legend(loc='upper right')
ax_ecg.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig_ecg)

# 2. Side-by-Side HRV Tachogram & PSD Spectrum Charts
st.subheader("Autonomic Nervous System & HRV Spectral Diagnostics")
col_left, col_right = st.columns(2)

with col_left:
    fig_tach, ax_tach = plt.subplots(figsize=(6, 3.2))
    ax_tach.plot(rr, color='#1f77b4', marker='o', markersize=2, linewidth=0.8)
    ax_tach.set_title("RR Interval Tachogram")
    ax_tach.set_xlabel("Beat Number")
    ax_tach.set_ylabel("RR Interval (ms)")
    ax_tach.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_tach)

with col_right:
    fig_psd, ax_psd = plt.subplots(figsize=(6, 3.2))
    ax_psd.plot(freqs, psd, color='#d62728', linewidth=1.8)
    ax_psd.fill_between(freqs, psd, where=((freqs >= 0.04) & (freqs < 0.15)), color='#ff7f0e', alpha=0.35, label=f'LF ({lf:.1f} ms²)')
    ax_psd.fill_between(freqs, psd, where=((freqs >= 0.15) & (freqs < 0.40)), color='#d62728', alpha=0.35, label=f'HF ({hf:.1f} ms²)')
    ax_psd.set_title(f"HRV Power Spectral Density (LF/HF Ratio: {ratio:.2f})")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power (ms²/Hz)")
    ax_psd.set_xlim(0, 0.5)
    ax_psd.legend(loc='upper right')
    ax_psd.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_psd)
import pandas as pd

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Clinical Report Export")

# Build beat-by-beat diagnostic table
beat_times = np.round(valid_peaks / fs, 3)
beat_labels = ["Arrhythmia / Ectopic (1)" if c == 1 else "Normal Sinus (0)" for c in beat_classes]

df_report = pd.DataFrame({
    "Beat_Number": np.arange(1, len(valid_peaks) + 1),
    "Sample_Index": valid_peaks,
    "Timestamp_Sec": beat_times,
    "AI_Classification": beat_labels
})

csv_data = df_report.to_csv(index=False).encode('utf-8')

st.sidebar.download_button(
    label=f"⬇️ Download Record {record_choice} Beat Log (CSV)",
    data=csv_data,
    file_name=f"ECG_Clinical_Report_Record_{record_choice}.csv",
    mime="text/csv"
)

summary_text = f"""==================================================
CARDIOVASCULAR AI & HRV DIAGNOSTIC REPORT
Patient Record: MIT-BIH #{record_choice}
==================================================
- Mean Heart Rate:        {60000 / np.mean(rr):.1f} BPM
- Total Beats Analyzed:   {len(peaks)}
- AI Arrhythmia Burden:   {num_abnormal} Beats ({burden_pct:.2f}%)
- SDNN (Time HRV):        {sdnn:.2f} ms
- RMSSD (Vagal Tone):     {rmssd:.2f} ms
- LF Power (0.04-0.15Hz): {lf:.2f} ms^2
- HF Power (0.15-0.40Hz): {hf:.2f} ms^2
- LF/HF Autonomic Ratio:  {ratio:.2f}
==================================================
"""

st.sidebar.download_button(
    label="📋 Download Summary Report (TXT)",
    data=summary_text,
    file_name=f"ECG_Summary_Record_{record_choice}.txt",
    mime="text/plain"
)