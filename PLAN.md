# Score Improvement Plan — SKAB HDC

Current: F1=0.71 | Target: ≥0.74 | Leaderboard best: Conv-AE 0.78

---

## Approaches (roughly by expected impact)

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
