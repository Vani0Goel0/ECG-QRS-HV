"""
Phase 2: Classical Pan-Tompkins DSP Pipeline (Corrected)
Implements the 5-stage Pan-Tompkins algorithm to filter raw ECG and detect R-peaks.
"""

import numpy as np
import scipy.signal as sp_signal
import matplotlib.pyplot as plt
from setup import load_ecg_record

def pan_tompkins_detector(ecg_signal, fs):
    """
    Applies the 5-stage Pan-Tompkins QRS detection algorithm with zero-phase 
    filtering and local R-peak crest alignment.
    """
    print("[*] Running Pan-Tompkins DSP Pipeline...")
    
    # --- Stage 1: Zero-Phase Bandpass Filter (5-15 Hz) ---
    lowcut = 5.0
    highcut = 15.0
    nyq = 0.5 * fs
    b, a = sp_signal.butter(1, [lowcut / nyq, highcut / nyq], btype='band')
    filtered_ecg = sp_signal.filtfilt(b, a, ecg_signal)
    
    # --- Stage 2: Derivative Filter ---
    derivative_ecg = np.gradient(filtered_ecg)
    
    # --- Stage 3: Squaring Function ---
    squared_ecg = derivative_ecg ** 2
    
    # --- Stage 4: Moving Window Integration ---
    window_width = int(0.15 * fs)
    integrated_ecg = np.convolve(squared_ecg, np.ones(window_width)/window_width, mode='same')
    
    # --- Stage 5: Adaptive Thresholding & Peak Detection ---
    min_distance = int(0.2 * fs)
    raw_peaks, _ = sp_signal.find_peaks(integrated_ecg, distance=min_distance, prominence=np.max(integrated_ecg)*0.1)
    
    # --- Stage 6: Local Crest Alignment (Snaps peaks to exact R-wave tip) ---
    search_radius = int(0.08 * fs)  # +/- 80ms search window
    aligned_peaks = []
    for p in raw_peaks:
        start = max(0, p - search_radius)
        end = min(len(ecg_signal), p + search_radius)
        true_crest = start + np.argmax(ecg_signal[start:end])
        aligned_peaks.append(true_crest)
        
    peaks = np.array(aligned_peaks)
    print(f"[+] DSP Pipeline complete. Detected and aligned {len(peaks)} R-peaks.")
    return filtered_ecg, integrated_ecg, peaks

def evaluate_performance(detected_peaks, true_peaks, tolerance_samples=36):
    """
    Evaluates detector accuracy against ground truth annotations.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    matched_gt = set()
    for det in detected_peaks:
        matches = np.where(np.abs(true_peaks - det) <= tolerance_samples)[0]
        if len(matches) > 0:
            matched_gt.add(matches[0])
            true_positives += 1
        else:
            false_positives += 1
            
    false_negatives = len(true_peaks) - len(matched_gt)
    
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    print(f"\n--- Performance Evaluation (Pan-Tompkins DSP) ---")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Sensitivity (Recall): {sensitivity * 100:.2f}%")
    print(f"Precision (Positive Predictivity): {precision * 100:.2f}%")
    print(f"F1-Score: {f1_score * 100:.2f}%")
    
    return sensitivity, precision, f1_score

def plot_dsp_stages(raw_signal, filtered, integrated, true_peaks, detected_peaks, fs, duration_sec=5):
    """
    Visualizes the transformation across the Pan-Tompkins pipeline.
    """
    samples = int(fs * duration_sec)
    time = np.arange(samples) / fs
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(time, raw_signal[:samples], color='#1f77b4', alpha=0.6, label='Raw ECG')
    axes[0].plot(time, filtered[:samples], color='#2ca02c', label='Bandpass Filtered (5-15 Hz)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    axes[1].plot(time, integrated[:samples], color='#ff7f0e', label='Moving Window Integration')
    window_detected = detected_peaks[detected_peaks < samples]
    axes[1].scatter(window_detected / fs, integrated[window_detected], color='red', zorder=5, label='Detected Peaks')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    axes[2].plot(time, raw_signal[:samples], color='black', alpha=0.5, label='Raw ECG')
    window_gt = true_peaks[true_peaks < samples]
    axes[2].scatter(window_gt / fs, raw_signal[window_gt], color='blue', marker='x', s=60, zorder=5, label='Ground Truth Peaks')
    axes[2].scatter(window_detected / fs, raw_signal[window_detected], color='red', marker='o', facecolors='none', s=80, zorder=6, label='Detected Peaks')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle("Pan-Tompkins DSP Pipeline Stages & Peak Detection Validation")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    target_record = '100'
    data_dir = 'mitdb_cache'
    
    ecg_signal, sampling_rate, true_annotations = load_ecg_record(record_name=target_record, db_dir=data_dir)
    
    filtered_signal, integrated_signal, computed_peaks = pan_tompkins_detector(ecg_signal, sampling_rate)
    
    evaluate_performance(computed_peaks, true_annotations)
    
    plot_dsp_stages(ecg_signal, filtered_signal, integrated_signal, true_annotations, computed_peaks, sampling_rate, duration_sec=5)