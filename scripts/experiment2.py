"""
HDC SKAB — Experiment 2: Temporal Binding + Multi-Scale
Usage: python scripts/experiment2.py
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DATA_DIR    = os.environ.get("SKAB_DATA_DIR",
              "/home/benj/repos/bjs-drones-1/data/skab/data")
SENSOR_COLS = ["Accelerometer1RMS","Accelerometer2RMS","Current","Pressure",
               "Temperature","Thermocouple","Voltage","Volume Flow RateRMS"]
N_SENSORS   = len(SENSOR_COLS)
TRAIN_SIZE  = 400
D           = 10_000

def load_files(data_dir):
    dfs = []
    for sub in ["valve1", "valve2", "other"]:
        path = os.path.join(data_dir, sub)
        if not os.path.exists(path): continue
        for f in sorted(os.listdir(path)):
            if f.endswith(".csv"):
                dfs.append(pd.read_csv(os.path.join(path, f), sep=";",
                                       index_col="datetime", parse_dates=True))
    return dfs

# ── raw window extraction (no feature engineering) ────────────────────────────
def raw_windows(df, start, end, window_size):
    """Returns (N, window_size, N_SENSORS) float32 array."""
    arr = df[SENSOR_COLS].values.astype(np.float32)
    out = []
    for i in range(start + window_size - 1, end):
        out.append(arr[max(start, i - window_size + 1):i + 1])
    if not out:
        return np.empty((0, window_size, N_SENSORS), dtype=np.float32)
    ## pad short windows at the start of the file
    padded = []
    for w in out:
        if len(w) < window_size:
            pad = np.tile(w[0], (window_size - len(w), 1))
            w = np.concatenate([pad, w], axis=0)
        padded.append(w)
    return np.array(padded, dtype=np.float32)

# ── stat feature extraction (current best) ───────────────────────────────────
def stat_features(arr):
    diff = np.diff(arr, axis=0)
    return np.concatenate([
        arr.mean(0), arr.std(0).clip(1e-9), arr.min(0), arr.max(0),
        np.abs(diff).mean(0), diff.std(0).clip(1e-9),
    ]).astype(np.float32)

def stat_windows(df, start, end, window_size):
    arr = df[SENSOR_COLS].values.astype(np.float32)
    return np.array([
        stat_features(arr[max(start, i - window_size + 1):i + 1])
        for i in range(start + window_size - 1, end)
    ])

# ── cross-sensor features (add to stat) ──────────────────────────────────────
def stat_cross_features(arr):
    base = stat_features(arr)
    ## pairwise Pearson correlations (upper triangle, 28 pairs)
    if arr.shape[0] < 2:
        corr_flat = np.zeros(N_SENSORS * (N_SENSORS - 1) // 2, dtype=np.float32)
    else:
        corr = np.corrcoef(arr.T)  ## (N_SENSORS, N_SENSORS)
        idx = np.triu_indices(N_SENSORS, k=1)
        corr_flat = corr[idx].astype(np.float32)
        corr_flat = np.nan_to_num(corr_flat, nan=0.0)
    return np.concatenate([base, corr_flat])

def stat_cross_windows(df, start, end, window_size):
    arr = df[SENSOR_COLS].values.astype(np.float32)
    return np.array([
        stat_cross_features(arr[max(start, i - window_size + 1):i + 1])
        for i in range(start + window_size - 1, end)
    ])

# ── baseline encoder (sign-RP on stat features) ───────────────────────────────
def make_rp(n_features, seed=42):
    return np.random.RandomState(seed).randn(n_features, D).astype(np.float32)

def encode_stat(feats_norm, W):
    return np.sign(feats_norm @ W)

# ── temporal binding encoder ──────────────────────────────────────────────────
def make_pos_hvs(window_size, seed=99):
    """Fixed random bipolar position vectors, shape (W, D)."""
    rng = np.random.RandomState(seed)
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32),
                      size=(window_size, D))

def make_sensor_proj(seed=42):
    """Random projection per sensor → each timestep value → D-dim."""
    return np.random.RandomState(seed).randn(N_SENSORS, D).astype(np.float32)

def encode_temporal(raw_wins, W_sensor, pos_hvs, sensor_mn, sensor_std):
    """
    raw_wins: (N, W, N_SENSORS)
    For each t: bind sign(norm_x[t] @ W_sensor) ⊗ pos_hvs[t]
    Bundle over t, binarize.
    """
    N, W, _ = raw_wins.shape
    norm = (raw_wins - sensor_mn) / sensor_std  ## (N, W, N_SENSORS)
    bundled = np.zeros((N, D), dtype=np.float32)
    for t in range(W):
        step_hv = np.sign(norm[:, t, :] @ W_sensor)  ## (N, D)
        bundled += step_hv * pos_hvs[t]               ## (N, D) elementwise
    return np.sign(bundled)

# ── multi-scale: build + score at multiple window sizes ───────────────────────
def build_proto(enc):
    proto = np.sign(enc.sum(0))
    return proto / (np.linalg.norm(proto) + 1e-9)

def cosine_sims(hvs, proto_n):
    norms = np.linalg.norm(hvs, axis=1, keepdims=True).clip(1e-9)
    return (hvs / norms) @ proto_n

def smooth(sims, k=3):
    return pd.Series(sims).rolling(k).median().bfill().values

def best_f1_threshold(all_true, all_sims, n=200):
    thresholds = np.linspace(all_sims.min(), all_sims.max(), n)
    return max(
        ((f1_score(all_true, (all_sims < t).astype(float), zero_division=0), t)
         for t in thresholds),
        key=lambda x: x[0]
    )

# ── eval functions ────────────────────────────────────────────────────────────
def eval_stat(files, window_size, feat_fn=stat_windows):
    """Baseline: stat features + sign-RP."""
    train = np.concatenate([feat_fn(df, 0, TRAIN_SIZE, window_size) for df in files])
    mn, std = train.mean(0), train.std(0).clip(1e-9)
    norm_train = (train - mn) / std
    W = make_rp(train.shape[1])
    proto_n = build_proto(encode_stat(norm_train, W))
    all_true, all_sims = [], []
    for df in files:
        feats = feat_fn(df, TRAIN_SIZE, len(df), window_size)
        if len(feats) == 0: continue
        hvs = encode_stat((feats - mn) / std, W)
        sims = smooth(cosine_sims(hvs, proto_n))
        n = len(sims)
        labels = df["anomaly"].values[TRAIN_SIZE + window_size - 1:
                                      TRAIN_SIZE + window_size - 1 + n]
        all_sims.extend(sims[:len(labels)])
        all_true.extend(labels)
    f1, _ = best_f1_threshold(np.array(all_true), np.array(all_sims))
    return f1

def eval_temporal(files, window_size):
    """Temporal binding: bind pos-HV ⊗ value-HV at each timestep."""
    raw_train = np.concatenate([raw_windows(df, 0, TRAIN_SIZE, window_size) for df in files])
    ## per-sensor normalization from training data
    flat = raw_train.reshape(-1, N_SENSORS)
    sensor_mn = flat.mean(0).astype(np.float32)
    sensor_std = flat.std(0).clip(1e-9).astype(np.float32)
    W_sensor = make_sensor_proj()
    pos_hvs  = make_pos_hvs(window_size)
    enc_train = encode_temporal(raw_train, W_sensor, pos_hvs, sensor_mn, sensor_std)
    proto_n = build_proto(enc_train)
    all_true, all_sims = [], []
    for df in files:
        raws = raw_windows(df, TRAIN_SIZE, len(df), window_size)
        if len(raws) == 0: continue
        hvs = encode_temporal(raws, W_sensor, pos_hvs, sensor_mn, sensor_std)
        sims = smooth(cosine_sims(hvs, proto_n))
        n = len(sims)
        labels = df["anomaly"].values[TRAIN_SIZE + window_size - 1:
                                      TRAIN_SIZE + window_size - 1 + n]
        all_sims.extend(sims[:len(labels)])
        all_true.extend(labels)
    f1, _ = best_f1_threshold(np.array(all_true), np.array(all_sims))
    return f1

def eval_multiscale(files, scales=(20, 70, 170)):
    """Multi-scale: combine cosine sims from prototypes at different window sizes."""
    ## build one prototype per scale
    protos = {}
    Ws = {}
    mns, stds = {}, {}
    for w in scales:
        train = np.concatenate([stat_windows(df, 0, TRAIN_SIZE, w) for df in files])
        mn, std = train.mean(0), train.std(0).clip(1e-9)
        mns[w], stds[w] = mn, std
        W = make_rp(train.shape[1], seed=w)   ## different seed per scale
        Ws[w] = W
        protos[w] = build_proto(encode_stat((train - mn) / std, W))
    all_true, all_sims_combined = [], []
    max_w = max(scales)
    for df in files:
        scale_sims = {}
        for w in scales:
            feats = stat_windows(df, TRAIN_SIZE, len(df), w)
            if len(feats) == 0:
                scale_sims[w] = np.array([])
                continue
            hvs = encode_stat((feats - mns[w]) / stds[w], Ws[w])
            scale_sims[w] = cosine_sims(hvs, protos[w])
        ## align lengths (largest window gives fewest samples)
        lens = [len(v) for v in scale_sims.values() if len(v) > 0]
        if not lens: continue
        min_len = min(lens)
        combined = np.mean([scale_sims[w][-min_len:] for w in scales
                            if len(scale_sims[w]) > 0], axis=0)
        sims = smooth(combined)
        n = len(sims)
        labels = df["anomaly"].values[TRAIN_SIZE + max_w - 1:
                                      TRAIN_SIZE + max_w - 1 + n]
        all_sims_combined.extend(sims[:len(labels)])
        all_true.extend(labels)
    f1, _ = best_f1_threshold(np.array(all_true), np.array(all_sims_combined))
    return f1

def eval_temporal_multiscale(files, scales=(20, 70, 170)):
    """Temporal binding at multiple scales, combined."""
    protos, W_sensors, pos_hvs_d, sensor_mns, sensor_stds = {}, {}, {}, {}, {}
    for w in scales:
        raw_train = np.concatenate([raw_windows(df, 0, TRAIN_SIZE, w) for df in files])
        flat = raw_train.reshape(-1, N_SENSORS)
        sensor_mns[w] = flat.mean(0).astype(np.float32)
        sensor_stds[w] = flat.std(0).clip(1e-9).astype(np.float32)
        W_sensors[w]  = make_sensor_proj(seed=w)
        pos_hvs_d[w]  = make_pos_hvs(w, seed=99 + w)
        enc = encode_temporal(raw_train, W_sensors[w], pos_hvs_d[w],
                              sensor_mns[w], sensor_stds[w])
        protos[w] = build_proto(enc)
    all_true, all_sims_combined = [], []
    max_w = max(scales)
    for df in files:
        scale_sims = {}
        for w in scales:
            raws = raw_windows(df, TRAIN_SIZE, len(df), w)
            if len(raws) == 0:
                scale_sims[w] = np.array([])
                continue
            hvs = encode_temporal(raws, W_sensors[w], pos_hvs_d[w],
                                  sensor_mns[w], sensor_stds[w])
            scale_sims[w] = cosine_sims(hvs, protos[w])
        lens = [len(v) for v in scale_sims.values() if len(v) > 0]
        if not lens: continue
        min_len = min(lens)
        combined = np.mean([scale_sims[w][-min_len:] for w in scales
                            if len(scale_sims[w]) > 0], axis=0)
        sims = smooth(combined)
        n = len(sims)
        labels = df["anomaly"].values[TRAIN_SIZE + max_w - 1:
                                      TRAIN_SIZE + max_w - 1 + n]
        all_sims_combined.extend(sims[:len(labels)])
        all_true.extend(labels)
    f1, _ = best_f1_threshold(np.array(all_true), np.array(all_sims_combined))
    return f1


if __name__ == "__main__":
    files = load_files(DATA_DIR)
    print(f"{len(files)} files loaded\n")

    configs = [
        ("baseline  stat w=170",                  lambda: eval_stat(files, 170)),
        ("stat+cross-sensor w=170",               lambda: eval_stat(files, 170, stat_cross_windows)),
        ("temporal-binding w=170",                lambda: eval_temporal(files, 170)),
        ("temporal-binding w=20",                 lambda: eval_temporal(files, 20)),
        ("temporal-binding w=70",                 lambda: eval_temporal(files, 70)),
        ("multiscale stat (20+70+170)",           lambda: eval_multiscale(files, (20, 70, 170))),
        ("multiscale stat (70+170)",              lambda: eval_multiscale(files, (70, 170))),
        ("multiscale temporal (20+70+170)",       lambda: eval_temporal_multiscale(files, (20, 70, 170))),
        ("multiscale temporal (70+170)",          lambda: eval_temporal_multiscale(files, (70, 170))),
        ("stat w=170 + temporal w=170 combined",  None),  ## handled inline below
    ]

    print(f"{'Config':<45}  {'F1':>6}  {'Time':>7}")
    print("-" * 62)

    results = {}
    for name, fn in configs[:-1]:
        t0 = time.perf_counter()
        f1 = fn()
        elapsed = time.perf_counter() - t0
        results[name] = f1
        marker = "  ◀" if f1 > 0.821 else ""
        print(f"{name:<45}  {f1:.4f}  {elapsed:>5.1f}s{marker}")

    ## combined: average stat-w170 sims + temporal-w170 sims
    print(f"\n--- combined stat + temporal at w=170 ---")
    t0 = time.perf_counter()

    ## stat
    train_s = np.concatenate([stat_windows(df, 0, TRAIN_SIZE, 170) for df in files])
    mn_s, std_s = train_s.mean(0), train_s.std(0).clip(1e-9)
    W_s = make_rp(train_s.shape[1])
    proto_s = build_proto(encode_stat((train_s - mn_s) / std_s, W_s))

    ## temporal
    raw_tr = np.concatenate([raw_windows(df, 0, TRAIN_SIZE, 170) for df in files])
    flat = raw_tr.reshape(-1, N_SENSORS)
    s_mn = flat.mean(0).astype(np.float32)
    s_std = flat.std(0).clip(1e-9).astype(np.float32)
    W_sen = make_sensor_proj()
    p_hvs = make_pos_hvs(170)
    enc_t = encode_temporal(raw_tr, W_sen, p_hvs, s_mn, s_std)
    proto_t = build_proto(enc_t)

    all_true, all_sims = [], []
    for df in files:
        feats = stat_windows(df, TRAIN_SIZE, len(df), 170)
        raws  = raw_windows(df, TRAIN_SIZE, len(df), 170)
        if len(feats) == 0 or len(raws) == 0: continue
        hvs_s = encode_stat((feats - mn_s) / std_s, W_s)
        hvs_t = encode_temporal(raws, W_sen, p_hvs, s_mn, s_std)
        sims = smooth(0.5 * cosine_sims(hvs_s, proto_s) +
                      0.5 * cosine_sims(hvs_t, proto_t))
        n = len(sims)
        labels = df["anomaly"].values[TRAIN_SIZE + 170 - 1:
                                      TRAIN_SIZE + 170 - 1 + n]
        all_sims.extend(sims[:len(labels)])
        all_true.extend(labels)

    f1, _ = best_f1_threshold(np.array(all_true), np.array(all_sims))
    elapsed = time.perf_counter() - t0
    marker = "  ◀" if f1 > 0.821 else ""
    print(f"{'stat+temporal combined w=170':<45}  {f1:.4f}  {elapsed:>5.1f}s{marker}")
