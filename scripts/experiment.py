"""
HDC SKAB — Experiment script
Vergleicht verschiedene Encoding-Strategien und Feature-Sets.
Usage: python scripts/experiment.py [--data-dir PATH]
"""
import os, sys, time, argparse
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

# ── data loading ──────────────────────────────────────────────────────────────
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

# ── feature extraction ────────────────────────────────────────────────────────
def extract_basic(arr):
    diff = np.diff(arr, axis=0)
    return np.concatenate([
        arr.mean(0), arr.std(0).clip(1e-9), arr.min(0), arr.max(0),
        np.abs(diff).mean(0), diff.std(0).clip(1e-9),
    ]).astype(np.float32)

def extract_fft(arr):
    diff = np.diff(arr, axis=0)
    fft_mag = np.abs(np.fft.rfft(arr, axis=0))
    fft_energy = fft_mag.sum(0)
    dom_freq = fft_mag[1:].argmax(0).astype(float)
    ## spectral entropy
    fft_prob = fft_mag / (fft_mag.sum(0, keepdims=True) + 1e-9)
    spec_entropy = -(fft_prob * np.log(fft_prob + 1e-9)).sum(0)
    ## low / mid / high band energy (split rfft into 3 bands)
    n_bins = fft_mag.shape[0]
    b = n_bins // 3
    low  = fft_mag[:b].sum(0)
    mid  = fft_mag[b:2*b].sum(0)
    high = fft_mag[2*b:].sum(0)
    return np.concatenate([
        arr.mean(0), arr.std(0).clip(1e-9), arr.min(0), arr.max(0),
        np.abs(diff).mean(0), diff.std(0).clip(1e-9),
        fft_energy, dom_freq, spec_entropy, low, mid, high,
    ]).astype(np.float32)

EXTRACTORS = {"basic": extract_basic, "fft": extract_fft}
N_FEATURES = {"basic": N_SENSORS * 6, "fft": N_SENSORS * 12}

def make_windows(df, start, end, window_size, extractor):
    arr = df[SENSOR_COLS].values.astype(np.float32)
    return np.array([
        extractor(arr[max(start, i-window_size+1):i+1])
        for i in range(start + window_size - 1, end)
    ])

# ── SSP encoding (no nengo needed) ───────────────────────────────────────────
def make_unitary(dim, seed):
    rng = np.random.RandomState(seed)
    a = rng.rand((dim - 1) // 2)
    sign = rng.choice((-1, +1), len(a))
    phi = sign * np.pi * (1e-3 + a * (1 - 2e-3))
    fv = np.zeros(dim, dtype="complex128")
    fv[0] = 1.0
    fv[1:(dim + 1) // 2] = np.cos(phi) + 1j * np.sin(phi)
    fv[-1:dim // 2:-1]   = np.conj(fv[1:(dim + 1) // 2])
    if dim % 2 == 0:
        fv[dim // 2] = 1.0
    return np.fft.fft(np.fft.ifft(fv).real)   ## pre-compute fft for fast power

def make_ssp_axes(n_features, dim, seed=42):
    rng = np.random.RandomState(seed)
    return [make_unitary(dim, int(rng.randint(1e6))) for _ in range(n_features)]

def ssp_encode_batch(feats_norm, axes_fv):
    """feats_norm: (N, F), returns (N, D) real"""
    D = len(axes_fv[0])
    out = np.zeros((len(feats_norm), D), dtype="complex128")
    for fi, ax in enumerate(axes_fv):
        vals = feats_norm[:, fi]          ## (N,)
        ## ax ** val = exp(val * log(ax)) in frequency domain
        ## ax is already fft of the axis vector
        log_ax = np.log(ax + 1e-30)
        out += np.exp(np.outer(vals, log_ax))   ## (N, D)
    return np.fft.ifft(out).real.astype(np.float32)

# ── level encoding ────────────────────────────────────────────────────────────
def make_level_vecs(n_levels, dim, seed=42):
    rng = np.random.RandomState(seed)
    vecs = np.zeros((n_levels, dim), dtype=np.float32)
    vecs[0] = rng.choice([-1.0, 1.0], dim)
    for i in range(1, n_levels):
        n_flip = dim // (2 * n_levels)
        vecs[i] = vecs[i-1].copy()
        idx = rng.choice(dim, n_flip, replace=False)
        vecs[i][idx] *= -1
    return vecs

N_LEVELS = 100
_level_vecs = make_level_vecs(N_LEVELS, D)

def level_encode_batch(feats_norm, level_vecs=_level_vecs):
    """feats_norm: (N, F), returns (N, D)"""
    n_levels = len(level_vecs)
    ## map normalized values to level index (clamp to [-4,4] → [0, n_levels-1])
    clipped = np.clip(feats_norm, -4, 4)
    idxs = ((clipped + 4) / 8 * (n_levels - 1)).astype(int)
    ## superposition of level vecs for each feature
    out = np.zeros((len(feats_norm), level_vecs.shape[1]), dtype=np.float32)
    for fi in range(feats_norm.shape[1]):
        out += level_vecs[idxs[:, fi]]
    return np.sign(out)

# ── random projection (baseline) ─────────────────────────────────────────────
def make_rp_matrix(n_features, dim, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n_features, dim).astype(np.float32)

def rp_encode_batch(feats_norm, W):
    return np.sign(feats_norm @ W)

# ── eval one config ───────────────────────────────────────────────────────────
def eval_config(files, feat_key, window_size, encode_fn, n_seeds=1):
    extractor = EXTRACTORS[feat_key]
    n_feat = N_FEATURES[feat_key]

    ## collect train windows
    train_windows = np.concatenate([
        make_windows(df, 0, TRAIN_SIZE, window_size, extractor)
        for df in files
    ])
    mn  = train_windows.mean(0)
    std = train_windows.std(0).clip(1e-9)
    norm_train = (train_windows - mn) / std

    all_f1s = []
    for seed in range(n_seeds):
        ## encode train
        if encode_fn == "rp":
            W = make_rp_matrix(n_feat, D, seed=seed)
            enc_train = rp_encode_batch(norm_train, W)
        elif encode_fn == "level":
            lvecs = make_level_vecs(N_LEVELS, D, seed=seed)
            enc_train = level_encode_batch(norm_train, lvecs)
        elif encode_fn == "ssp":
            axes = make_ssp_axes(n_feat, D, seed=seed)
            enc_train = ssp_encode_batch(norm_train, axes)

        ## prototype
        proto = np.sign(enc_train.sum(0))
        proto_n = proto / (np.linalg.norm(proto) + 1e-9)

        ## inference
        all_true, all_sims = [], []
        for df in files:
            feats = make_windows(df, TRAIN_SIZE, len(df), window_size, extractor)
            if len(feats) == 0: continue
            norm_feats = (feats - mn) / std

            if encode_fn == "rp":
                hvs = rp_encode_batch(norm_feats, W)
            elif encode_fn == "level":
                hvs = level_encode_batch(norm_feats, lvecs)
            elif encode_fn == "ssp":
                hvs = ssp_encode_batch(norm_feats, axes)

            norms = np.linalg.norm(hvs, axis=1, keepdims=True).clip(1e-9)
            sims = (hvs / norms) @ proto_n
            smoothed = pd.Series(sims).rolling(3).median().bfill().values
            n = len(smoothed)
            labels = df["anomaly"].values[
                TRAIN_SIZE + window_size - 1:
                TRAIN_SIZE + window_size - 1 + n
            ]
            all_sims.extend(smoothed[:len(labels)])
            all_true.extend(labels)

        all_true_np = np.array(all_true, dtype=float)
        all_sims_np = np.array(all_sims, dtype=float)

        ## ensemble: accumulate sims across seeds, threshold once at end
        all_f1s.append(all_sims_np)
        _all_true = all_true_np

    ## ensemble: mean sims across seeds
    mean_sims = np.mean(all_f1s, axis=0)
    best_f1, _ = max(
        ((f1_score(_all_true, (mean_sims < t).astype(float), zero_division=0), t)
         for t in np.linspace(mean_sims.min(), mean_sims.max(), 200)),
        key=lambda x: x[0]
    )
    return best_f1


# ── experiment grid ───────────────────────────────────────────────────────────
CONFIGS = [
    {"name": "baseline (sign-RP, basic, w=20)",   "feat": "basic", "win": 20, "enc": "rp",    "seeds": 1},
    {"name": "+FFT features  (w=20)",              "feat": "fft",   "win": 20, "enc": "rp",    "seeds": 1},
    {"name": "+FFT +window30",                     "feat": "fft",   "win": 30, "enc": "rp",    "seeds": 1},
    {"name": "+FFT +window50",                     "feat": "fft",   "win": 50, "enc": "rp",    "seeds": 1},
    {"name": "+FFT w=30 +Level-Enc",               "feat": "fft",   "win": 30, "enc": "level", "seeds": 1},
    {"name": "+FFT w=30 +SSP-Enc",                 "feat": "fft",   "win": 30, "enc": "ssp",   "seeds": 1},
    {"name": "+FFT w=30 +Ensemble(5)",             "feat": "fft",   "win": 30, "enc": "rp",    "seeds": 5},
    {"name": "+FFT w=30 +Level +Ensemble(5)",      "feat": "fft",   "win": 30, "enc": "level", "seeds": 5},
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    print(f"Loading data from {args.data_dir}...")
    files = load_files(args.data_dir)
    print(f"{len(files)} files loaded\n")

    print(f"{'Config':<45}  {'F1':>6}  {'Time':>8}")
    print("-" * 65)
    for cfg in CONFIGS:
        t0 = time.perf_counter()
        f1 = eval_config(files,
                         feat_key=cfg["feat"],
                         window_size=cfg["win"],
                         encode_fn=cfg["enc"],
                         n_seeds=cfg["seeds"])
        elapsed = time.perf_counter() - t0
        marker = "  ◀ best" if f1 >= 0.74 else ""
        print(f"{cfg['name']:<45}  {f1:.4f}  {elapsed:>6.1f}s{marker}")
