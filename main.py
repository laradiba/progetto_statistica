# main.py

import sys
import pandas as pd
import numpy as np
from pathlib import Path

from baseline import run_baseline
from user_based_cf import run_usercf
from tuning import run_optuna


# =========================
# 0) SCELTA DATASET DA TERMINALE
# =========================
if len(sys.argv) < 2:
    print("Uso: python main.py [movielens | goodbooks]")
    sys.exit(1)

DATASET = sys.argv[1].lower()


# =========================
# 1) LOAD DATA
# =========================
if DATASET == "movielens":
    DATA_DIR = Path("data/ml-1m")

    ratings = pd.read_csv(
        DATA_DIR / "ratings.dat",
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        DATA_DIR / "movies.dat",
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1"
    )

elif DATASET == "goodbooks":

    DATA_DIR = Path("data/archive")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    ratings.columns = ratings.columns.str.strip()

    ratings = ratings.rename(columns={
        "user_id": "userId",
        "book_id": "movieId",
        "rating": "rating"
    })[["userId", "movieId", "rating"]].copy()

    # per compatibilità col codice MovieLens
    ratings["timestamp"] = 0

    # Goodbooks: books.csv ha colonna id (o book_id) e title
    books = pd.read_csv(DATA_DIR / "books.csv")
    books.columns = books.columns.str.strip()

    movies = books.rename(columns={"id": "movieId"})[["movieId", "title"]].copy()
    movies["genres"] = ""

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    movies["movieId"] = movies["movieId"].astype(int)


else:
    print("Dataset non riconosciuto. Usa 'movielens' o 'goodbooks'.")
    sys.exit(1)


# =========================
# 1.5) FILTRO UTENTI (evita crash nello split 1 test + 1 val)
# =========================
min_ratings = 3
counts = ratings.groupby("userId").size()
ratings = ratings[ratings["userId"].isin(counts[counts >= min_ratings].index)].copy()

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
