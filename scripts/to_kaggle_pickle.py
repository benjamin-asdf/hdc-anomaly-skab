"""
Convert Clojure-exported JSON predictions to SKAB pickle format for Kaggle submission.
Usage: python3 scripts/to_kaggle_pickle.py data/skab-predictions.json
"""
import json
import pickle
import sys
import pandas as pd

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)

predicted_outlier = [pd.Series(d["pred"], dtype=float) for d in data]

out = path.replace(".json", "-kaggle.pkl")
with open(out, "wb") as f:
    pickle.dump(predicted_outlier, f)

print(f"Wrote {out} ({len(predicted_outlier)} series)")
print("Upload this file to the Kaggle SKAB competition notebook.")
