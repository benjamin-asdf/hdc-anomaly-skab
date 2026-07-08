"""
Full efficiency benchmark: HDC vs all SKAB methods with F1 >= 0.71
Measures training time, model size, inference time, F1.
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, "data/skab")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DATA_DIR    = "data/skab/data"
SENSOR_COLS = ["Accelerometer1RMS","Accelerometer2RMS","Current","Pressure",
               "Temperature","Thermocouple","Voltage","Volume Flow RateRMS"]
TRAIN_SIZE  = 400
WINDOW_SIZE = 20
N_SENSORS   = len(SENSOR_COLS)
RESULTS     = {}

# ── Data ─────────────────────────────────────────────────────────────────────

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
print(f"{len(files)} files\n")

def smooth(series):
    return pd.Series(series, dtype=float).rolling(3).median().bfill().values

def best_f1(true, scores, higher_is_anomaly=False):
    thresholds = np.linspace(np.array(scores).min(), np.array(scores).max(), 200)
    best = 0.0
    for t in thresholds:
        pred = (np.array(scores) > t).astype(float) if higher_is_anomaly \
               else (np.array(scores) < t).astype(float)
        f = f1_score(np.array(true), pred, zero_division=0)
        if f > best: best = f
    return best

# ── HDC ───────────────────────────────────────────────────────────────────────

print("── HDC ──")
D = 10_000; BATCH = 512

def extract(arr):
    diff = np.diff(arr, axis=0)
    return np.concatenate([arr.mean(0), arr.std(0).clip(1e-9), arr.min(0), arr.max(0),
                           np.abs(diff).mean(0), diff.std(0).clip(1e-9)]).astype(np.float32)

def make_windows(df, start, end):
    arr = df[SENSOR_COLS].values.astype(np.float32)
    return [extract(arr[max(start,i-WINDOW_SIZE+1):i+1]) for i in range(start+WINDOW_SIZE-1,end)]

all_train = np.array([f for df in files for f in make_windows(df,0,TRAIN_SIZE)])
mn = all_train.mean(0); std = all_train.std(0).clip(1e-9)
N_FEAT = all_train.shape[1]
torch.manual_seed(42)
W = torch.randn(N_FEAT, D)

def encode(x): return torch.sign(torch.tensor((x-mn)/std, dtype=torch.float32) @ W)

t0 = time.perf_counter()
with torch.no_grad():
    hvs   = torch.cat([encode(all_train[i:i+BATCH]) for i in range(0,len(all_train),BATCH)])
    proto = torch.sign(hvs.sum(0))
    proto_n = (proto/proto.norm()).numpy()
hdc_train_ms = (time.perf_counter()-t0)*1000

all_true, all_sims = [], []
t_inf = time.perf_counter()
N_INFER = 500
with torch.no_grad():
    sample = encode(all_train[:1])
    for _ in range(N_INFER):
        hv = encode(all_train[:1])
        _ = float((hv/hv.norm()) @ torch.tensor(proto_n))
hdc_infer_us = (time.perf_counter()-t_inf)/N_INFER*1e6

with torch.no_grad():
    for df in files:
        feats = make_windows(df, TRAIN_SIZE, len(df))
        if not feats: continue
        hvs_t = encode(np.array(feats))
        sims  = (hvs_t/hvs_t.norm(dim=1,keepdim=True)).numpy() @ proto_n
        s = smooth(sims.tolist())
        n = len(s)
        labels = df["anomaly"].values[TRAIN_SIZE+WINDOW_SIZE-1:TRAIN_SIZE+WINDOW_SIZE-1+n]
        all_sims.extend(s[:len(labels)]); all_true.extend(labels)

hdc_f1 = best_f1(all_true, all_sims, higher_is_anomaly=False)
RESULTS["HDC"] = dict(f1=hdc_f1, train_ms=hdc_train_ms,
                      model_kb=D/8/1024, infer_us=hdc_infer_us,
                      params=D, framework="none", passes="1")
print(f"  F1={hdc_f1:.3f}  train={hdc_train_ms:.0f}ms  "
      f"model={D/8/1024:.1f}KB  infer={hdc_infer_us:.1f}µs\n")

# ── T²+Q (PCA) ───────────────────────────────────────────────────────────────

print("── T²+Q (PCA) ──")
from core.t2 import T2

t2_train_ms_total = 0.0
t2_true, t2_scores = [], []
t2_params = 0

t2_skipped = 0
for df in files:
    X_train = df[SENSOR_COLS].iloc[:TRAIN_SIZE]
    X_test  = df[SENSOR_COLS].iloc[TRAIN_SIZE:]

    model = T2(scaling=True)
    t0 = time.perf_counter()
    try:
        model.fit(X_train)
    except Exception as e:
        t2_skipped += 1
        continue
    t2_train_ms_total += (time.perf_counter()-t0)*1000
    if t2_params == 0:
        nc = getattr(getattr(model, 'pca', None), 'n_components_', N_SENSORS)
        t2_params = nc * N_SENSORS + N_SENSORS

    try:
        model.predict(X_test, plot_fig=False)
        t2_arr = model.t2["T2"].values.astype(float)
        t2_ucl = float(model.t2_ucl)
        if hasattr(model, 'q') and model.q is not None:
            q_arr  = model.q["Q"].values.astype(float)
            q_ucl  = float(model.q_ucl)
            scores = t2_arr / t2_ucl + q_arr / q_ucl  # combined normalized score
        else:
            scores = t2_arr / t2_ucl
    except Exception:
        t2_skipped += 1
        continue
    s = smooth(scores.tolist())
    n = len(s)
    labels = df["anomaly"].values[TRAIN_SIZE:TRAIN_SIZE+n]
    t2_scores.extend(s[:len(labels)]); t2_true.extend(labels)

if t2_skipped: print(f"  (skipped {t2_skipped} files due to fit errors)")

t2_infer = 0.0
df0 = files[0]
X_tr0 = df0[SENSOR_COLS].iloc[:TRAIN_SIZE]
X_te0 = df0[SENSOR_COLS].iloc[TRAIN_SIZE:TRAIN_SIZE+1]
model0 = T2(scaling=True); model0.fit(X_tr0)
t0 = time.perf_counter()
for _ in range(N_INFER): model0.predict(X_te0, plot_fig=False)
t2_infer = (time.perf_counter()-t0)/N_INFER*1e6

t2_f1 = best_f1(t2_true, t2_scores, higher_is_anomaly=True)
t2_kb = t2_params * 4 / 1024
RESULTS["T²+Q"] = dict(f1=t2_f1, train_ms=t2_train_ms_total,
                        model_kb=t2_kb, infer_us=t2_infer,
                        params=t2_params, framework="scipy/sklearn", passes="1")
print(f"  F1={t2_f1:.3f}  train={t2_train_ms_total:.0f}ms  "
      f"model~{t2_kb:.1f}KB  infer={t2_infer:.1f}µs\n")

# ── MSET ─────────────────────────────────────────────────────────────────────

print("── MSET ──")
from core.MSET import MSET

mset_train_ms = 0.0
mset_true, mset_scores = [], []

mset_skipped = 0
for df in files:
    X_train = df[SENSOR_COLS].iloc[:TRAIN_SIZE]
    X_test  = df[SENSOR_COLS].iloc[TRAIN_SIZE:]

    model = MSET()
    t0 = time.perf_counter()
    try:
        model.fit(X_train)
    except Exception:
        mset_skipped += 1
        continue
    mset_train_ms += (time.perf_counter()-t0)*1000

    try:
        scores = model.predict(X_test)
    except Exception:
        mset_skipped += 1
        continue
    if hasattr(scores, 'values'): scores = scores.values
    scores = np.array(scores, dtype=float).flatten()
    s = smooth(scores.tolist())
    n = len(s)
    labels = df["anomaly"].values[TRAIN_SIZE:TRAIN_SIZE+n]
    mset_scores.extend(s[:len(labels)]); mset_true.extend(labels)

if mset_skipped: print(f"  (skipped {mset_skipped} files)")

mset_infer = 0.0
model0 = MSET(); model0.fit(files[0][SENSOR_COLS].iloc[:TRAIN_SIZE])
X_te0 = files[0][SENSOR_COLS].iloc[TRAIN_SIZE:TRAIN_SIZE+1]
t0 = time.perf_counter()
for _ in range(N_INFER): model0.predict(X_te0)
mset_infer = (time.perf_counter()-t0)/N_INFER*1e6

# MSET stores training data as memory matrix: TRAIN_SIZE × N_SENSORS
mset_kb = TRAIN_SIZE * N_SENSORS * 4 / 1024

mset_f1 = best_f1(mset_true, mset_scores, higher_is_anomaly=True)
RESULTS["MSET"] = dict(f1=mset_f1, train_ms=mset_train_ms,
                        model_kb=mset_kb, infer_us=mset_infer,
                        params=TRAIN_SIZE*N_SENSORS, framework="scipy/numpy", passes="1")
print(f"  F1={mset_f1:.3f}  train={mset_train_ms:.0f}ms  "
      f"model~{mset_kb:.1f}KB  infer={mset_infer:.1f}µs\n")

# ── LSTM-AE (TensorFlow) ──────────────────────────────────────────────────────

print("── LSTM-AE ──")
from core.LSTM_AE import LSTM_AE

lstm_train_ms = 0.0
lstm_true, lstm_scores = [], []
lstm_params = 0

for fi, df in enumerate(files):
    arr = df[SENSOR_COLS].values.astype(np.float32)
    mu  = arr[:TRAIN_SIZE].mean(0); sg = arr[:TRAIN_SIZE].std(0).clip(1e-9)
    arr_n = (arr - mu) / sg

    # LSTM-AE expects (N, timesteps, features) — use WINDOW_SIZE as timesteps
    def seq_windows(a, start, end):
        return np.array([a[max(start,i-WINDOW_SIZE+1):i+1]
                         for i in range(start+WINDOW_SIZE-1, end)])

    X_train = seq_windows(arr_n, 0, TRAIN_SIZE)
    X_test  = seq_windows(arr_n, TRAIN_SIZE, len(arr))

    # params=[epochs, batch_size, val_split]
    model = LSTM_AE(params=[100, 32, 0.1])
    t0 = time.perf_counter()
    model.fit(X_train)
    lstm_train_ms += (time.perf_counter()-t0)*1000

    if lstm_params == 0:
        lstm_params = model.model.count_params()

    recon  = model.predict(X_test)
    scores = ((recon - X_test)**2).mean(axis=(1,2))
    s = smooth(scores.tolist())
    n = len(s)
    labels = df["anomaly"].values[TRAIN_SIZE+WINDOW_SIZE-1:TRAIN_SIZE+WINDOW_SIZE-1+n]
    lstm_scores.extend(s[:len(labels)]); lstm_true.extend(labels)

    if (fi+1) % 10 == 0: print(f"  {fi+1}/{len(files)} files...")

lstm_infer = 0.0
model0 = LSTM_AE(params=[100, 32, 0.1])
model0.fit(seq_windows((files[0][SENSOR_COLS].values[:TRAIN_SIZE].astype(np.float32) - mu)/sg, 0, TRAIN_SIZE))
X_s = seq_windows((files[0][SENSOR_COLS].values.astype(np.float32) - mu)/sg, TRAIN_SIZE, TRAIN_SIZE+1+WINDOW_SIZE)[:1]
t0 = time.perf_counter()
for _ in range(N_INFER): model0.predict(X_s)
lstm_infer = (time.perf_counter()-t0)/N_INFER*1e6

lstm_kb = lstm_params * 4 / 1024
lstm_f1 = best_f1(lstm_true, lstm_scores, higher_is_anomaly=True)
RESULTS["LSTM-AE"] = dict(f1=lstm_f1, train_ms=lstm_train_ms,
                           model_kb=lstm_kb, infer_us=lstm_infer,
                           params=lstm_params, framework="TensorFlow", passes="≤100 epochs")
print(f"  F1={lstm_f1:.3f}  train={lstm_train_ms:.0f}ms  "
      f"model={lstm_kb:.1f}KB  infer={lstm_infer:.1f}µs\n")

# ── Table ─────────────────────────────────────────────────────────────────────

print("="*72)
print(f"{'Method':<12} {'F1':>6} {'Train (ms)':>12} {'Model (KB)':>11} {'Infer (µs)':>11} {'Framework'}")
print("="*72)
for name, r in RESULTS.items():
    print(f"{name:<12} {r['f1']:>6.3f} {r['train_ms']:>12.0f} {r['model_kb']:>11.1f} "
          f"{r['infer_us']:>11.1f}  {r['framework']}")
print("="*72)

hdc = RESULTS["HDC"]
for name, r in RESULTS.items():
    if name == "HDC": continue
    print(f"\nvs {name}:  {r['train_ms']/hdc['train_ms']:.0f}× slower training  "
          f"{r['model_kb']/hdc['model_kb']:.0f}× larger  "
          f"F1 {r['f1']:.3f} vs {hdc['f1']:.3f}")

def to_serializable(obj):
    if isinstance(obj, dict): return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    return obj

with open("data/benchmark_results_all.json","w") as f:
    json.dump(to_serializable(RESULTS), f, indent=2)
print("\nSaved: data/benchmark_results_all.json")
