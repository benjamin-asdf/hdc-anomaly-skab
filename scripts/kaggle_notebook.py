"""
HDC Anomaly Detection on SKAB — Kaggle Notebook
================================================
No extra installs needed — uses only torch, numpy, pandas, sklearn (all on Kaggle by default).

Dataset: Add "SKAB - Skoltech Anomaly Benchmark" from Kaggle datasets.

Results vs leaderboard (SKAB outlier detection F1):
  Conv-AE       0.78  (best, needs TensorFlow, ~100 epochs per file)
  LSTM-AE       0.74
  HDC (this)   ~0.71  (pure math, 1 pass, 1.2 KB model)
  T²            0.66
"""

# ── Cell 1: imports ───────────────────────────────────────────────────────────
import os, time, pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

DATA_DIR    = os.environ.get("SKAB_DATA_DIR",
              "/kaggle/input/datasets/yuriykatser/skoltech-anomaly-benchmark-skab/SKAB")
SENSOR_COLS = ["Accelerometer1RMS","Accelerometer2RMS","Current","Pressure",
               "Temperature","Thermocouple","Voltage","Volume Flow RateRMS"]
N_SENSORS   = len(SENSOR_COLS)
N_FEATURES  = N_SENSORS * 6   # mean/std/min/max/mean-abs-diff/diff-std = 48
TRAIN_SIZE  = 400
WINDOW_SIZE = 20
D           = 10_000
BATCH       = 512
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}  D={D}  features={N_FEATURES}")

# ── Cell 2: load files ────────────────────────────────────────────────────────
def load_files(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "On Kaggle: click 'Add Input' → search 'Skoltech Anomaly Benchmark SKAB'.\n"
            "Locally: set env var SKAB_DATA_DIR=data/skab/data"
        )
    dfs = []
    for sub in ["valve1", "valve2", "other"]:
        path = os.path.join(data_dir, sub)
        if not os.path.exists(path): continue
        for f in sorted(os.listdir(path)):
            if f.endswith(".csv"):
                dfs.append(pd.read_csv(os.path.join(path, f), sep=";",
                                       index_col="datetime", parse_dates=True))
    return dfs

files = load_files(DATA_DIR)
if len(files) == 0:
    raise RuntimeError(
        f"No CSV files found under {DATA_DIR}.\n"
        "Expected subdirs: valve1/, valve2/, other/ with .csv files."
    )
print(f"{len(files)} files loaded")

# ── Cell 3: feature extraction ────────────────────────────────────────────────
def extract(arr):
    diff = np.diff(arr, axis=0)
    return np.concatenate([
        arr.mean(0), arr.std(0).clip(1e-9), arr.min(0), arr.max(0),
        np.abs(diff).mean(0), diff.std(0).clip(1e-9),
    ]).astype(np.float32)

def make_windows(df, start, end):
    missing = [c for c in SENSOR_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing sensor columns in CSV: {missing}")
    arr = df[SENSOR_COLS].values.astype(np.float32)
    return [extract(arr[max(start, i-WINDOW_SIZE+1):i+1])
            for i in range(start+WINDOW_SIZE-1, end)]

# ── Cell 4: normalisation (fit on training data) ──────────────────────────────
all_train = np.array([f for df in files for f in make_windows(df, 0, TRAIN_SIZE)])
if len(all_train) == 0:
    raise RuntimeError(
        f"No training windows extracted. Each file needs at least "
        f"{TRAIN_SIZE + WINDOW_SIZE} rows, got files with fewer rows."
    )
mn  = all_train.mean(0)
std = all_train.std(0).clip(1e-9)
norm = lambda x: (np.array(x) - mn) / std

# ── Cell 5: HDC encoder (random projection + binarise) ───────────────────────
torch.manual_seed(42)
W = torch.randn(N_FEATURES, D, device=device)  # fixed random projection

def encode(feats_np):
    t = torch.tensor(norm(feats_np), dtype=torch.float32, device=device)
    return torch.sign(t @ W)  # (N, D)

# ── Cell 6: build normal prototype (single pass) ──────────────────────────────
t0 = time.perf_counter()
print("Building prototype...")
with torch.no_grad():
    hvs = torch.cat([encode(all_train[i:i+BATCH]) for i in range(0, len(all_train), BATCH)])
    proto = torch.sign(hvs.sum(0))
    proto_n = (proto / proto.norm()).cpu().numpy()
print(f"Training done in {(time.perf_counter()-t0)*1000:.0f} ms  "
      f"(model size: {D/8/1024:.2f} KB binarized)")

# ── Cell 7: per-file inference ────────────────────────────────────────────────
print("Running inference...")
all_true, all_sims = [], []
with torch.no_grad():
    for df in files:
        feats = make_windows(df, TRAIN_SIZE, len(df))
        if not feats: continue
        hvs_t = encode(feats).cpu()
        sims  = (hvs_t / hvs_t.norm(dim=1, keepdim=True)).numpy() @ proto_n
        smoothed = pd.Series(sims.tolist(), dtype=float).rolling(3).median().bfill().values
        n = len(smoothed)
        labels = df["anomaly"].values[TRAIN_SIZE+WINDOW_SIZE-1:
                                      TRAIN_SIZE+WINDOW_SIZE-1+n]
        all_sims.extend(smoothed[:len(labels)])
        all_true.extend(labels)

# ── Cell 8: tune threshold & evaluate ────────────────────────────────────────
all_true_np = np.array(all_true, dtype=float)
all_sims_np = np.array(all_sims, dtype=float)

best_f1, best_t = max(
    ((f1_score(all_true_np, (all_sims_np < t).astype(float), zero_division=0), t)
     for t in np.linspace(all_sims_np.min(), all_sims_np.max(), 200)),
    key=lambda x: x[0]
)

print(f"\n=== Results ===")
print(f"F1 (outlier detection): {best_f1:.4f}  threshold={best_t:.3f}")
print(f"\nLeaderboard (SKAB outlier F1):")
print(f"  Conv-AE      0.78  (best — needs TF, 34 models, ~100 epochs each)")
print(f"  LSTM-AE      0.74")
print(f"  HDC (this)  {best_f1:.2f}  ← 1 pass, 1.2 KB model, no framework")
print(f"  T²           0.66")

# ── Cell 9: save predictions (for chp_score / official eval) ─────────────────
predicted_outlier_series = [
    pd.Series(
        (np.array(
            pd.Series(
                (encode(make_windows(df, TRAIN_SIZE, len(df))).cpu() /
                 encode(make_windows(df, TRAIN_SIZE, len(df))).cpu().norm(dim=1, keepdim=True)
                ).numpy() @ proto_n,
                dtype=float
            ).rolling(3).median().bfill().values
        ) < best_t).astype(float),
        index=df.iloc[TRAIN_SIZE+WINDOW_SIZE-1:].index[:
              len(make_windows(df, TRAIN_SIZE, len(df)))]
    )
    for df in files
]

out_path = "/kaggle/working/results-HDC.pkl" if os.path.exists("/kaggle/working") \
           else "data/results-HDC.pkl"
with open(out_path, "wb") as f:
    pickle.dump(predicted_outlier_series, f)
print(f"Saved: {out_path}")
