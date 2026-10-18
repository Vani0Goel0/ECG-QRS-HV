"""
Phase 4: Heart Rate Variability (HRV) Analysis Engine
Computes time-domain (SDNN, RMSSD) and frequency-domain (LF/HF ratio) metrics from R-peaks.
Run: pip install numpy scipy matplotlib
"""

import numpy as np
import scipy.signal as sp_signal
import matplotlib.pyplot as plt
from setup import load_ecg_record

def compute_rr_intervals(r_peaks, fs):
    """
    Computes normal-to-normal (NN) RR intervals in milliseconds from R-peak sample indices.
    """
    # Difference between consecutive R-peaks in samples
    rr_samples = np.diff(r_peaks)
    
    # Convert samples to milliseconds: (samples / sampling_rate) * 1000
    rr_ms = (rr_samples / fs) * 1000.0
    return rr_ms

def compute_time_domain_hrv(rr_ms):
    """
    Computes time-domain HRV metrics:
    - SDNN: Standard deviation of NN intervals (reflects overall autonomic activity)
    - RMSSD: Root mean square of successive differences (reflects parasympathetic activity)
    """
    sdnn = np.std(rr_ms, ddof=1)
    
    # Successive differences between adjacent RR intervals
    successive_diffs = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2))
    
    return sdnn, rmssd

def compute_frequency_domain_hrv(rr_ms, fs_resample=4.0):
    """
    Computes frequency-domain HRV metrics using the Lomb-Scargle Periodogram or FFT:
    - LF (Low Frequency, 0.04 - 0.15 Hz): Sympathetic & Parasympathetic activity
    - HF (High Frequency, 0.15 - 0.40 Hz): Parasympathetic activity (respiratory)
    - LF/HF Ratio: Autonomic sympathovagal balance
    """
    # Create time axis for RR intervals in seconds
    rr_time = np.cumsum(rr_ms) / 1000.0
    rr_time = rr_time - rr_time[0]  # Start at t = 0
    
    # Remove ectopic beats or outliers if necessary, then interpolate to a uniform time grid
    uniform_time = np.arange(rr_time[0], rr_time[-1], 1.0 / fs_resample)
    interpolated_rr = np.interp(uniform_time, rr_time, rr_ms)
    
    # Remove mean (detrend) for spectral analysis
    detrended_rr = interpolated_rr - np.mean(interpolated_rr)
    
    # Compute Power Spectral Density (PSD) using Welch's method
    frequencies, psd = sp_signal.welch(detrended_rr, fs=fs_resample, nperseg=256)
    
    # Define frequency bands
    lf_band = (frequencies >= 0.04) & (frequencies < 0.15)
    hf_band = (frequencies >= 0.15) & (frequencies < 0.40)
    
    # Integrate power under the curve for each band
    lf_power = np.trapz(psd[lf_band], frequencies[lf_band])
    hf_power = np.trapz(psd[hf_band], frequencies[hf_band])
    
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0
    
    return frequencies, psd, lf_power, hf_power, lf_hf_ratio

def plot_hrv_results(rr_ms, frequencies, psd):
    """
    Plots the RR interval tachogram and the Frequency-Domain Power Spectral Density.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    # Tachogram
    axes[0].plot(rr_ms, color='#1f77b4', marker='o', markersize=3, linestyle='-')
    axes[0].set_title("RR Interval Tachogram (Heart Rate Variability)")
    axes[0].set_xlabel("Beat Number")
    axes[0].set_ylabel("RR Interval (ms)")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Power Spectral Density
    axes[1].plot(frequencies, psd, color='#d62728', linewidth=2)
    axes[1].fill_between(frequencies, psd, where=((frequencies >= 0.04) & (frequencies < 0.40)), color='#d62728', alpha=0.3)
    axes[1].set_title("HRV Power Spectral Density (PSD)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power ($ms^2/Hz$)")
    axes[1].set_xlim(0, 0.5)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("[*] Loading ECG record for HRV analysis...")
    ecg_signal, fs, r_peaks = load_ecg_record(record_name='100', db_dir='mitdb_cache')
    
    # Step 1: Compute RR Intervals
    rr_intervals_ms = compute_rr_intervals(r_peaks, fs)
    
    # Step 2: Compute Time-Domain Metrics
    sdnn, rmssd = compute_time_domain_hrv(rr_intervals_ms)
    
    # Step 3: Compute Frequency-Domain Metrics
    freqs, spectrum, lf, hf, ratio = compute_frequency_domain_hrv(rr_intervals_ms)
    
    print("\n" + "="*40)
    print("      CLINICAL HRV DIAGNOSTIC REPORT      ")
    print("="*40)
    print(f"Total Beats Analyzed : {len(r_peaks)}")
    print(f"Mean Heart Rate      : {60000 / np.mean(rr_intervals_ms):.1f} BPM")
    print("-" * 40)
    print(f"Time-Domain Metrics:")
    print(f"  • SDNN             : {sdnn:.2f} ms")
    print(f"  • RMSSD            : {rmssd:.2f} ms")
    print("-" * 40)
    print(f"Frequency-Domain Metrics:")
    print(f"  • LF Power         : {lf:.2f} ms²")
    print(f"  • HF Power         : {hf:.2f} ms²")
    print(f"  • LF/HF Ratio      : {ratio:.2f}")
    print("="*40)
    
    # Step 4: Visualize Results
    plot_hrv_results(rr_intervals_ms, freqs, spectrum)