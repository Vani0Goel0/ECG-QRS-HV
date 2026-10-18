"""
Phase 3: Real Arrhythmia 1D-CNN Classifier (Normal vs. Ectopic/Arrhythmia)
Self-contained: Works seamlessly with any version of setup.py!
"""

import os
import wfdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from setup import load_ecg_record

MODEL_PATH = "ecg_cnn_weights.pth"

# Clinical AAMI Beat Mapping: 'N' is Normal (0), Ectopic/Arrhythmic beats are Abnormal (1)
NORMAL_SYMBOLS = {'N', '.'}

def load_record_with_symbols(rec, db_dir='mitdb_cache'):
    """
    Loads ECG signal and R-peaks via setup.py, and fetches clinical beat symbols 
    ('N', 'V', 'A', etc.) with local caching.
    """
    signal, fs, r_peaks = load_ecg_record(record_name=str(rec), db_dir=db_dir)
    
    sym_cache_file = os.path.join(db_dir, f"{rec}_symbols.npy")
    local_atr = os.path.join(db_dir, f"{rec}.atr")
    
    if os.path.exists(sym_cache_file):
        symbols = np.load(sym_cache_file, allow_pickle=True)
    elif os.path.exists(local_atr):
        ann = wfdb.rdann(os.path.join(db_dir, str(rec)), 'atr')
        symbols = np.array(ann.symbol)
        np.save(sym_cache_file, symbols)
    else:
        print(f"[*] Fetching clinical arrhythmia symbols for Record {rec} from PhysioNet...")
        ann = wfdb.rdann(str(rec), 'atr', pn_dir='mitdb')
        symbols = np.array(ann.symbol)
        np.save(sym_cache_file, symbols)
        
    return signal, fs, r_peaks, symbols

class ECGWindowDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.signals)
        
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

class Lightweight1DCNN(nn.Module):
    def __init__(self, input_length=180):
        super(Lightweight1DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        flattened_size = 32 * (input_length // 4)
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2) # Class 0: Normal, Class 1: Arrhythmia (PVC/PAC/Bundle Branch)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def extract_labeled_epochs(ecg_signal, r_peaks, symbols=None, fs=360, window_size_ms=250):
    """Extracts 180-sample Z-normalized windows around each R-peak."""
    samples_half = int((window_size_ms / 1000.0) * fs)
    epochs = []
    labels = []
    valid_peaks = []
    
    # Snap ground-truth/detected peaks to exact local R-wave crest (+/- 80ms)
    radius = int(0.08 * fs)
    
    for idx, p in enumerate(r_peaks):
        s_search = max(0, p - radius)
        e_search = min(len(ecg_signal), p + radius)
        peak = s_search + np.argmax(ecg_signal[s_search:e_search])
        
        start = peak - samples_half
        end = peak + samples_half
        if start >= 0 and end < len(ecg_signal):
            window = ecg_signal[start:end]
            if len(window) == 2 * samples_half:
                window = (window - np.mean(window)) / (np.std(window) + 1e-8)
                epochs.append(window)
                valid_peaks.append(peak)
                
                if symbols is not None and idx < len(symbols):
                    sym = symbols[idx]
                    labels.append(0 if sym in NORMAL_SYMBOLS else 1)
                else:
                    labels.append(0)
                    
    return np.array(epochs), np.array(labels), np.array(valid_peaks)

def train_and_save_model(db_dir='mitdb_cache'):
    """Trains the 1D-CNN on Record 100 (Normal) + Record 106 (High PVCs) and saves weights."""
    print("[*] Building balanced training set from Record 100 & Record 106...")
    X_list, y_list = [], []
    
    for rec in ['100', '106']:
        sig, fs, peaks, syms = load_record_with_symbols(rec, db_dir=db_dir)
        ep, lbl, _ = extract_labeled_epochs(sig, peaks, syms, fs)
        X_list.append(ep)
        y_list.append(lbl)
        
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_loader = DataLoader(ECGWindowDataset(X_train, y_train), batch_size=32, shuffle=True)
    
    model = Lightweight1DCNN(input_length=X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"[*] Training 1D-CNN on {len(X_train)} beats ({np.sum(y_train==1)} arrhythmic beats)...")
    for epoch in range(5):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch [{epoch+1}/5] - Loss: {total_loss/len(train_loader):.4f}")
        
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)), dim=1).numpy()
    print("\n--- 1D-CNN Arrhythmia Classification Report ---")
    print(classification_report(y_test, preds, target_names=['Normal (N)', 'Arrhythmia (PVC/Ectopic)']))
    
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[+] Saved trained PyTorch weights to '{MODEL_PATH}'.")
    return model

def predict_arrhythmias(ecg_signal, detected_peaks, fs=360):
    """Runs trained 1D-CNN inference on detected R-peaks for the Streamlit dashboard."""
    epochs, _, valid_peaks = extract_labeled_epochs(ecg_signal, detected_peaks, None, fs)
    if len(epochs) == 0:
        return np.array([]), np.array([])
        
    model = Lightweight1DCNN(input_length=epochs.shape[1])
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        tensor_x = torch.tensor(epochs, dtype=torch.float32).unsqueeze(1)
        logits = model(tensor_x)
        predictions = torch.argmax(logits, dim=1).numpy()
        
    return valid_peaks, predictions

if __name__ == "__main__":
    train_and_save_model()