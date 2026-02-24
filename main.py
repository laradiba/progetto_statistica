# main.py

import pandas as pd
import numpy as np
from pathlib import Path

from baseline import run_baseline
from user_based_cf import run_usercf
from tuning import run_optuna

# =========================
# 1) LOAD DATA
# =========================
DATA_DIR = Path("data/ml-1m")
ratings_path = DATA_DIR / "ratings.dat"
movies_path = DATA_DIR / "movies.dat"

ratings = pd.read_csv(
    ratings_path,
    sep="::",
    engine="python",
    names=["userId", "movieId", "rating", "timestamp"]
)

movies = pd.read_csv(
    movies_path,
    sep="::",
    engine="python",
    names=["movieId", "title", "genres"],
    encoding="latin-1"
)

ratings = ratings.sample(frac=1, random_state=42).reset_index(drop=True)

# =========================
# 2) SPLIT (1 test + 1 val per utente)
# =========================
rng = np.random.default_rng(42)

holdout_idx = (
    ratings
    .groupby("userId")["movieId"]
    .apply(lambda x: rng.choice(x.index, size=2, replace=False))
    .explode()
    .astype(int)
)

holdout_idx = holdout_idx.to_numpy().reshape(-1, 2)

test_idx = holdout_idx[:, 0]
val_idx  = holdout_idx[:, 1]

test = ratings.loc[test_idx].copy()
val  = ratings.loc[val_idx].copy()
train = ratings.drop(np.concatenate([test_idx, val_idx])).copy()

print("Train:", train.shape)
print("Val:", val.shape)
print("Test:", test.shape)

# =========================
# 3) RUN MODELS
# =========================

print("\n===== BASELINE =====")
run_baseline(train, val, test)


print("\n===== USER-BASED CF =====")
run_usercf(train, val, test)

print("\n===== OPTUNA TUNING (MF) =====")
run_optuna(train, val, test, ratings, movies, n_trials=5)

