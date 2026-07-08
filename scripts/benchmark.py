"""
Efficiency benchmark: HDC vs Conv-AE on SKAB
Measures: training time, model size, inference time, F1
"""

import os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

DATA_DIR    = "data/skab/data"
SENSOR_COLS = ["Accelerometer1RMS","Accelerometer2RMS","Current","Pressure",
               "Temperature","Thermocouple","Voltage","Volume Flow RateRMS"]
TRAIN_SIZE  = 400
N_SENSORS   = len(SENSOR_COLS)

# ── Data loading ─────────────────────────────────────────────────────────────

def load_files(data_dir):
    dfs = []
    for sub in ["valve1","valve2","other"]:
        path = os.path.join(data_dir, sub)
        if not os.path.exists(path): continue
        for f in sorted(os.listdir(path)):
            if f.endswith(".csv"):
                dfs.append(pd.read_csv(os.path.join(path,f), sep=";",
                                       index_col="datetime", parse_dates=True))
    return dfs

files = load_files(DATA_DIR)
print(f"Files: {len(files)}")

# ── Shared feature extraction ─────────────────────────────────────────────────

WINDOW_SIZE = 20

def make_windows(arr, start, end):
    out = []
    for i in range(start + WINDOW_SIZE - 1, end):
        w = arr[max(start, i - WINDOW_SIZE + 1):i + 1]
        feats = np.concatenate([w.mean(0), w.std(0).clip(1e-9),
                                w.min(0), w.max(0),
                                np.abs(np.diff(w, axis=0)).mean(0),
                                np.diff(w, axis=0).std(0).clip(1e-9)])
        out.append(feats.astype(np.float32))
    return np.array(out)  # (T, 48)

all_train = np.concatenate([
    make_windows(df[SENSOR_COLS].values.astype(np.float32), 0, TRAIN_SIZE)
    for df in files
])
mn = all_train.mean(0); std = all_train.std(0).clip(1e-9)
norm = lambda x: (x - mn) / std

# ── HDC ───────────────────────────────────────────────────────────────────────

D = 10_000
N_FEAT = all_train.shape[1]

torch.manual_seed(42)
W = torch.randn(N_FEAT, D)   # random projection matrix — fixed

def hdc_encode_batch(x_np):
    t = torch.tensor(norm(x_np), dtype=torch.float32)
    return torch.sign(t @ W)  # (N, D)

print("\n── HDC ──")
t0 = time.perf_counter()
BATCH = 512
with torch.no_grad():
    hvs = torch.cat([hdc_encode_batch(all_train[i:i+BATCH])
                     for i in range(0, len(all_train), BATCH)])
    proto = torch.sign(hvs.sum(0))                     # (D,)
    proto_n = (proto / proto.norm()).numpy()
hdc_train_time = time.perf_counter() - t0

# Model size: projection matrix W + prototype vector
# W is fixed random (not stored in prod — can be regenerated from seed)
# Prototype: D × 1 bit = 1.25 KB binarized, or D × float32 = 40 KB
hdc_model_params = D          # just the prototype
hdc_model_kb_binary = D / 8 / 1024          # binarized bits → KB
hdc_model_kb_float  = D * 4 / 1024          # float32 → KB

# Inference timing
test_sample = make_windows(files[0][SENSOR_COLS].values.astype(np.float32),
                           TRAIN_SIZE, len(files[0]))
t0 = time.perf_counter()
N_INFER = 1000
for _ in range(N_INFER):
    with torch.no_grad():
        hv = hdc_encode_batch(test_sample[:1])
        _ = float((hv / hv.norm()) @ torch.tensor(proto_n))
hdc_infer_us = (time.perf_counter() - t0) / N_INFER * 1e6

# F1
all_true, all_sims = [], []
with torch.no_grad():
    for df in files:
        feats = make_windows(df[SENSOR_COLS].values.astype(np.float32),
                             TRAIN_SIZE, len(df))
        if len(feats) == 0: continue
        hvs_t = hdc_encode_batch(feats)
        sims = (hvs_t / hvs_t.norm(dim=1, keepdim=True)).numpy() @ proto_n
        smoothed = pd.Series(sims.tolist(), dtype=float).rolling(3).median().bfill().values
        labels = df["anomaly"].values[TRAIN_SIZE+WINDOW_SIZE-1:
                                      TRAIN_SIZE+WINDOW_SIZE-1+len(smoothed)]
        all_sims.extend(smoothed[:len(labels)]); all_true.extend(labels)

all_true_np = np.array(all_true, dtype=float)
all_sims_np = np.array(all_sims, dtype=float)
hdc_f1, hdc_t = max(
    ((f1_score(all_true_np, (all_sims_np < t).astype(float), zero_division=0), t)
     for t in np.linspace(all_sims_np.min(), all_sims_np.max(), 200)),
    key=lambda x: x[0]
)

print(f"  Training time : {hdc_train_time*1000:.1f} ms  (single pass, {len(all_train)} windows)")
print(f"  Model size    : {hdc_model_kb_binary:.2f} KB (binarized) / {hdc_model_kb_float:.1f} KB (float32)")
print(f"  Inference     : {hdc_infer_us:.1f} µs / window")
print(f"  F1            : {hdc_f1:.4f}")

# ── Conv-AE (PyTorch, matches SKAB architecture) ─────────────────────────────

# SKAB Conv-AE uses sequences of shape (time_steps, n_sensors) fed as (batch, time, ch)
SEQ_LEN = WINDOW_SIZE

class ConvAE(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(n_ch, 32, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 16, kernel_size=7, stride=2, padding=3), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=7, stride=2, padding=3,
                               output_padding=1), nn.ReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose1d(32, 32, kernel_size=7, stride=2, padding=3,
                               output_padding=1), nn.ReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose1d(32, n_ch, kernel_size=7, padding=3),
        )
    def forward(self, x):  # x: (B, C, T)
        return self.dec(self.enc(x))

def raw_windows(arr, start, end, seq_len):
    out = []
    for i in range(start + seq_len - 1, end):
        out.append(arr[max(start, i - seq_len + 1):i + 1])
    return np.array(out, dtype=np.float32)  # (T, seq_len, n_ch)

print("\n── Conv-AE ──")
cae_total_time = 0.0
cae_params = sum(p.numel() for p in ConvAE(N_SENSORS).parameters())
cae_model_kb = cae_params * 4 / 1024

cae_all_true, cae_scores = [], []

for fi, df in enumerate(files):
    arr = df[SENSOR_COLS].values.astype(np.float32)
    # Per-sensor z-score on training window
    tr  = arr[:TRAIN_SIZE]
    mu  = tr.mean(0); sg = tr.std(0).clip(1e-9)
    arr_n = (arr - mu) / sg

    train_seq = raw_windows(arr_n, 0, TRAIN_SIZE, SEQ_LEN)  # (N, T, C)
    test_seq  = raw_windows(arr_n, TRAIN_SIZE, len(arr), SEQ_LEN)

    model = ConvAE(N_SENSORS)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.MSELoss()

    # (N, C, T) for Conv1d
    X_tr = torch.tensor(train_seq).permute(0, 2, 1)
    X_te = torch.tensor(test_seq).permute(0, 2, 1)

    t0 = time.perf_counter()
    best_loss = float("inf"); patience = 0
    for epoch in range(100):
        model.train()
        idx = torch.randperm(len(X_tr))
        ep_loss = 0.0
        for i in range(0, len(X_tr), 32):
            b = X_tr[idx[i:i+32]]
            opt.zero_grad()
            loss = crit(model(b), b)
            loss.backward(); opt.step()
            ep_loss += loss.item()
        ep_loss /= max(1, len(X_tr) // 32)
        # Early stopping on train loss (no val split — keeps it comparable)
        if ep_loss < best_loss - 1e-4:
            best_loss = ep_loss; patience = 0
        else:
            patience += 1
            if patience >= 5: break
    cae_total_time += time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        recon = model(X_te)
        scores = ((recon - X_te)**2).mean(dim=(1,2)).numpy()

    smoothed = pd.Series(scores.tolist(), dtype=float).rolling(3).median().bfill().values
    labels   = df["anomaly"].values[TRAIN_SIZE+SEQ_LEN-1:
                                    TRAIN_SIZE+SEQ_LEN-1+len(smoothed)]
    cae_scores.extend(smoothed[:len(labels)]); cae_all_true.extend(labels)
    if (fi+1) % 10 == 0:
        print(f"  {fi+1}/{len(files)} files done...")

# Inference timing (single window)
model_single = ConvAE(N_SENSORS)
model_single.eval()
sample_seq = torch.tensor(test_seq[:1]).permute(0, 2, 1)
t0 = time.perf_counter()
for _ in range(N_INFER):
    with torch.no_grad(): _ = model_single(sample_seq)
cae_infer_us = (time.perf_counter() - t0) / N_INFER * 1e6

cae_true_np  = np.array(cae_all_true, dtype=float)
cae_score_np = np.array(cae_scores,   dtype=float)
cae_f1, _    = max(
    ((f1_score(cae_true_np, (cae_score_np > t).astype(float), zero_division=0), t)
     for t in np.linspace(cae_score_np.min(), cae_score_np.max(), 200)),
    key=lambda x: x[0]
)

print(f"  Training time : {cae_total_time*1000:.0f} ms  ({len(files)} models × up-to-100 epochs)")
print(f"  Parameters    : {cae_params:,}  → {cae_model_kb:.1f} KB")
print(f"  Inference     : {cae_infer_us:.1f} µs / window")
print(f"  F1            : {cae_f1:.4f}")

# ── Comparison table ──────────────────────────────────────────────────────────

print("\n" + "="*62)
print(f"{'Metric':<28} {'HDC':>12} {'Conv-AE':>12}")
print("="*62)
print(f"{'F1 (SKAB outlier)':<28} {hdc_f1:>12.3f} {cae_f1:>12.3f}")
print(f"{'Training time (ms)':<28} {hdc_train_time*1000:>12.1f} {cae_total_time*1000:>12.0f}")
print(f"{'Model size (KB)':<28} {hdc_model_kb_binary:>12.2f} {cae_model_kb:>12.1f}")
print(f"{'Inference (µs/window)':<28} {hdc_infer_us:>12.1f} {cae_infer_us:>12.1f}")
print(f"{'Training passes':<28} {'1 (single pass)':>12} {'≤100 epochs':>12}")
print(f"{'Framework required':<28} {'none':>12} {'PyTorch/TF':>12}")
print(f"{'Incremental learning':<28} {'yes (bundle)':>12} {'retrain':>12}")
print("="*62)

speedup_train = cae_total_time / hdc_train_time
size_ratio    = cae_model_kb / hdc_model_kb_binary
infer_ratio   = cae_infer_us / hdc_infer_us
f1_ratio      = hdc_f1 / cae_f1

print(f"\nHDC vs Conv-AE:")
print(f"  {speedup_train:.0f}× faster training")
print(f"  {size_ratio:.0f}× smaller model")
print(f"  {infer_ratio:.1f}× faster inference")
print(f"  {f1_ratio:.0%} of Conv-AE F1")

# Save for HTML report
results = {
    "hdc":    {"f1": hdc_f1, "train_ms": hdc_train_time*1000,
               "model_kb": hdc_model_kb_binary, "infer_us": hdc_infer_us,
               "params": hdc_model_params},
    "conv_ae":{"f1": cae_f1, "train_ms": cae_total_time*1000,
               "model_kb": cae_model_kb, "infer_us": cae_infer_us,
               "params": cae_params},
}
with open("data/benchmark_results.json","w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: data/benchmark_results.json")
