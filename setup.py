"""
Phase 1: Ultra-Fast Data Loader & Local NPZ Cache for ECG Project
Eliminates slow wfdb.dl_database() calls by streaming directly and caching compressed NumPy arrays.
"""

import os
import wfdb
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Standard clinical records in the MIT-BIH Arrhythmia Database
MITBIH_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

def get_cached_records(db_dir='mitdb_cache'):
    """Returns a list of records already saved locally on disk."""
    if not os.path.exists(db_dir):
        return []
    cached = []
    for rec in MITBIH_RECORDS:
        npz_path = os.path.join(db_dir, f"{rec}.npz")
        hea_path = os.path.join(db_dir, f"{rec}.hea")
        if os.path.exists(npz_path) or os.path.exists(hea_path):
            cached.append(rec)
    return cached

def load_ecg_record(record_name='100', db_dir='mitdb_cache'):
    """
    Loads an ECG record in milliseconds from a local .npz cache.
    If not cached yet, streams directly from PhysioNet (bypassing slow dl_database)
    and saves as .npz so you never have to download it again.
    """
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        
    rec_str = str(record_name)
    npz_file = os.path.join(db_dir, f"{rec_str}.npz")
    local_hea = os.path.join(db_dir, f"{rec_str}.hea")
    
    # 1. FASTEST PATH: Load compressed NumPy binary (< 0.01 seconds)
    if os.path.exists(npz_file):
        data = np.load(npz_file)
        return data['signal'], int(data['fs']), data['r_peaks']
        
    # 2. SECOND PATH: Convert existing local .dat/.hea files (like Record 100) into .npz
    if os.path.exists(local_hea):
        record_path = os.path.join(db_dir, rec_str)
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
    else:
        # 3. FAST STREAMING PATH: Stream directly from PhysioNet without dl_database overhead
        print(f"[*] Fast-streaming record '{rec_str}' from PhysioNet and caching locally...")
        record = wfdb.rdrecord(rec_str, pn_dir='mitdb')
        annotation = wfdb.rdann(rec_str, 'atr', pn_dir='mitdb')
        
    signal = record.p_signal[:, 0].astype(np.float32)
    fs = int(record.fs)
    r_peak_indices = annotation.sample
    
    # Save as compressed .npz so future loads are instant
    np.savez_compressed(npz_file, signal=signal, fs=fs, r_peaks=r_peak_indices)
    print(f"[+] Record '{rec_str}' permanently cached at '{npz_file}'.")
    
    return signal, fs, r_peak_indices

def prefetch_records_parallel(records_to_fetch, db_dir='mitdb_cache'):
    """Downloads and caches multiple records simultaneously using multi-threading."""
    print(f"[*] Pre-fetching {len(records_to_fetch)} records in parallel...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        for rec in records_to_fetch:
            executor.submit(load_ecg_record, rec, db_dir)
    print("[+] All selected records are now cached locally for instant offline access!")

if __name__ == "__main__":
    # Run `python setup.py` once to parallel-cache the first 10 records in ~5-10 seconds
    starter_batch = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109']
    prefetch_records_parallel(starter_batch, db_dir='mitdb_cache')