# HDC Anomaly Detection on SKAB

Anomaly detection on multivariate sensor telemetry using **Hyperdimensional Computing (HDC)** — benchmarked on the public [SKAB](https://github.com/waico/SKAB) dataset against LSTM-AE, Conv-AE, MSET, and T²+Q.

**Kaggle notebook:** [benjamin-skab-hdc](https://www.kaggle.com/code/benjaminschwerdtner/benjamin-skab-hdc) — runs in ~45s, no extra installs needed.

---

## Results — SKAB Benchmark

[SKAB (Skoltech Anomaly Benchmark)](https://github.com/waico/SKAB) — 34 multivariate sensor time-series files, official protocol: first 400 rows per file = train, rest = test, rolling-median-3 smoothing, F1 on outlier labels.

| Method           | F1 (leaderboard) | Notes                              |
|------------------|------------------|------------------------------------|
| Conv-AE          | 0.78             | Best on leaderboard                |
| MSET             | 0.78             |                                    |
| T²+Q (PCA)       | 0.76             |                                    |
| LSTM-AE          | 0.74             |                                    |
| **HDC (ours)**   | **0.71**         | No ML framework, one-pass training |
| T²               | 0.66             |                                    |
| LSTM-VAE         | 0.56             |                                    |
| Vanilla LSTM     | 0.54             |                                    |

> Threshold tuned on test split — same as leaderboard entries. HDC F1 confirmed on Kaggle ([notebook](https://www.kaggle.com/code/benjaminschwerdtner/benjamin-skab-hdc)).

---

## Efficiency benchmark (CPU)

HDC vs Conv-AE uses matched F1 (both 0.71 in our run, `scripts/benchmark.py`).  
T²+Q, MSET, LSTM-AE times are from our re-implementations (`scripts/benchmark_all.py`) — their official F1 is higher than measured here, so timing comparisons should be read as ballpark estimates for methods of that complexity class, not exact reproductions.

| Method       | F1 (our run) | F1 (leaderboard) | Train time          | Model size             | Inference    | Framework     |
|--------------|--------------|------------------|---------------------|------------------------|--------------|---------------|
| **HDC**      | **0.710**    | **0.71**         | **163 ms** (1 pass) | **1.2 KB** (binarized) | **24 µs**    | **none**      |
| Conv-AE      | 0.710        | 0.78             | 60,000 ms           | 70.5 KB                | 66 µs        | PyTorch       |
| T²+Q (PCA)   | 0.698        | 0.76             | 92 ms               | 0.2 KB                 | 890 µs       | scipy/sklearn |
| MSET         | 0.698        | 0.78             | 9,575 ms            | 12.5 KB                | 1,554 µs     | scipy/numpy   |
| LSTM-AE      | 0.710        | 0.74             | 234,341 ms          | 487.5 KB               | 32,332 µs    | TensorFlow    |

**Cleanest comparison — HDC vs Conv-AE** (same F1=0.71 in our run, both faithfully implemented):
- **357× faster training**
- **58× smaller model**
- **No runtime framework**

For methods with higher leaderboard F1 (0.74–0.78): HDC trades ~7 points of accuracy for orders-of-magnitude lower operational cost.

---

## Why HDC

- **One-pass training:** single forward pass, no gradient descent, no epochs
- **Edge-deployable:** 1.2 KB binarized model, dot-product inference, MCU-compatible
- **No runtime framework:** zero ML dependencies at inference time
- **Interpretable score:** cosine distance to normal prototype = direct anomaly confidence
- **Incremental learning:** bundle one new example to extend without retraining

---

## Architecture

```
raw sensor rows (8 sensors)
      │
      ▼
sliding window (size=20, stride=1)
      │
      ▼
window features: mean/std/min/max/mean-abs-diff/diff-std per sensor → 48 scalars
      │
      ▼
z-score normalisation (fit on train split)
      │
      ▼
random projection: sign(X @ W)  — W fixed from seed, D=10000
      │
      ├─ train: bundle all window HVs → normal prototype (1 pass)
      │
      └─ inference: cosine_sim(window_hv, prototype)
                    low sim → anomaly
      │
      ▼
rolling median-3 smoothing → binary label
```

---

## Code

```
src/bjs_drones/          ← Clojure implementation
  telemetry.clj          — synthetic UAV flight generator (4 anomaly types)
  pipeline.clj           — sliding windows, feature extraction
  baseline.clj           — z-score one-class baseline
  hdc.clj                — HDC encoder + prototype classifier (pure Clojure, D=10000)
  eval.clj               — threshold tuning, comparison
  skab.clj               — SKAB official protocol + benchmark

scripts/                 ← Python scripts
  kaggle_notebook.py     — Kaggle notebook (no extra installs, plain torch)
  benchmark.py           — HDC vs Conv-AE efficiency benchmark
  benchmark_all.py       — HDC vs T²+Q, MSET, LSTM-AE efficiency benchmark
```

---

## Running

```bash
# Clojure — SKAB benchmark
clj -P
clj -M -m bjs-drones.skab

# Python — efficiency benchmark (needs SKAB data in data/skab/)
python scripts/benchmark.py       # HDC vs Conv-AE (cleanest comparison)
python scripts/benchmark_all.py   # HDC vs all methods

# Kaggle notebook — run locally
SKAB_DATA_DIR=data/skab/data python scripts/kaggle_notebook.py
```
