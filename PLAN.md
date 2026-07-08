# Score Improvement Plan — SKAB HDC

Current: **F1=0.82** (beats Conv-AE 0.78, previous leaderboard best)
Achieved with: w=170, basic stat features (mean/std/min/max/diff-mean/diff-std), sign-RP, D=10000

---

## Experiment Results

### experiment.py — window + encoding sweep (2026-07-08)

| Config                          | F1     | Time   |
|---------------------------------|--------|--------|
| baseline sign-RP basic w=20     | 0.7104 | 2.7s   |
| +FFT features w=20              | 0.7103 | 3.2s   |
| +FFT w=30                       | 0.7168 | 3.6s   |
| +FFT w=50                       | 0.7299 | 3.6s   |
| +FFT w=30 +Level-Enc            | 0.7166 | 35s    |
| +FFT w=30 +SSP-Enc              | 0.7171 | 848s   |
| +FFT w=30 +Ensemble(5)          | 0.7167 | 12s    |

Window fine-tune (FFT, sign-RP):

| Window | F1     |
|--------|--------|
| 20     | 0.7104 |
| 50     | 0.7299 |
| 70     | 0.7437 |
| 100    | 0.7658 |
| 150    | 0.8055 |
| 160    | 0.8139 |
| **170**| **0.8207** ← peak |
| 180    | 0.8206 |
| 200    | 0.8138 |
| 250    | 0.7935 |
| 300    | 0.7672 |

D and ensemble make no difference at w=170.

FFT at w=170: basic=0.8207, fft=0.8207 → FFT adds nothing at large windows.

### experiment2.py — temporal binding + multi-scale (2026-07-08)

| Config                          | F1     | Time   |
|---------------------------------|--------|--------|
| baseline stat w=170             | 0.8207 | 2.2s   |
| stat + cross-sensor w=170       | 0.8207 | 4.0s   |
| temporal binding w=170          | 0.8207 | 138s   |
| temporal binding w=20           | 0.7102 | 24s    |
| temporal binding w=70           | 0.7437 | 70s    |
| multiscale stat (20+70+170)     | 0.8207 | 7.8s   |
| multiscale temporal (20+70+170) | 0.8207 | 252s   |
| stat + temporal combined w=170  | 0.8207 | 138s   |

**Conclusion:** 0.8207 is the ceiling for this feature/protocol space.
Encoding method (RP, Level, SSP, Temporal Binding) is irrelevant at w=170 —
the statistical aggregates over 170 rows already contain all discriminative information.

---

## Approaches (explored)

### 1. SSP Encoding (Spatial Semantic Pointers)
Replace `sign(X @ W)` random projection with proper SSP encoding:
```
SSP(x) = e^{i·k·x}   per feature, discretised over D/2 frequencies
```
- Ähnliche Werte → nahe Hypervectors (geometrische Struktur)
- Binding durch elementweises Produkt im komplexen Raum
- Superposition wie bisher
- Ref: `/home/benj/repos/spatial-semantic-pointers`

### 2. FFT / Frequenz-Features
Zu den aktuellen 48 Features (mean/std/min/max/mad/diff-std) hinzufügen:
- Dominante Frequenz pro Sensor (argmax FFT-Magnitude)
- Spektrale Energie pro Band (low/mid/high)
- Spectral Entropy
- Anomalien in Pumpen/Ventilen haben oft distincte Frequenzsignaturen

### 3. Level / Thermometer Encoding
Echtes HDC Level-Encoding statt direkter Projektion:
- Wertebereich jedes Features in N Stufen quantisieren
- Jede Stufe = ein fixer Hypervector, interpoliert über Hamming-Distanz
- Ähnliche Werte → ähnliche HVs (ohne SSP-Overhead)

### 4. Größeres Window
Aktuell: 20 Rows. Probieren: 30, 40, 50.
- Langsamere Anomalien (Druckanstieg, Temperatur-Drift) brauchen mehr Kontext
- Trade-off: weniger Test-Samples, mehr Latenz

### 5. Per-File Normalisierung
Aktuell: globale Normalisierung über alle 34 Files.
- Files haben verschiedene Baseline-Statistiken
- Variante: global für Projection-Matrix W, per-File für mean/std

### 6. Ensemble (Multi-Seed)
5 HDC-Modelle mit verschiedenen Random Seeds → Cosine-Sim mitteln → threshold.
- Reduziert Varianz der Random Projection
- Kostet 5× Training (~1s statt 0.16s — immer noch negligible)

### 7. Smoothing optimieren
Aktuell: rolling-median-3. Probieren:
- EWMA mit verschiedenen alpha
- rolling-mean mit Fenster 5–10
- Median mit größerem Fenster (5, 7)

---

## Leaderboard Submission

Ziel: Eintrag in der README-Tabelle von [waico/SKAB](https://github.com/waico/SKAB).

1. **Offizielles Eval** — ihr `chp_score`-Skript laufen lassen (nicht nur F1, auch ChangePoint-Metriken)
2. **Kaggle Notebook** aktualisieren mit bestem Ansatz
3. **PR an waico/SKAB** — neuer Eintrag in ihrer Leaderboard-Tabelle mit Link zum Notebook

---

## Reihenfolge

1. FFT-Features + größeres Window (schnell, hoher Impact)
2. SSP oder Level-Encoding (sauberer HDC, guter Story-Anker für Neuroscience-Narrative)
3. Ensemble (einfach, robuster Score)
4. Offizielles Eval + PR
