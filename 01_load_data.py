import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/ml-1m")  # cambia se il tuo percorso è diverso

ratings_path = DATA_DIR / "ratings.dat"
movies_path  = DATA_DIR / "movies.dat"
users_path   = DATA_DIR / "users.dat"

print("ratings:", ratings_path.exists(), ratings_path)
print("movies:", movies_path.exists(), movies_path)
print("users:", users_path.exists(), users_path)
