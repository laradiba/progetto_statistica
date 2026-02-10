import pandas as pd
import numpy as np
from pathlib import Path

# === PATHS ===
DATA_DIR = Path("data/ml-1m")

ratings_path = DATA_DIR / "ratings.dat"
movies_path  = DATA_DIR / "movies.dat"
users_path   = DATA_DIR / "users.dat"
# === LOAD DATA ===

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

users = pd.read_csv(
    users_path,
    sep="::",
    engine="python",
    names=["userId", "gender", "age", "occupation", "zip"]
)

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)
print("Users shape:", users.shape)
