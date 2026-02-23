import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

def run_usercf(train, val, test):

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    # =====================
    # RE-INDEX USERS/MOVIES
    # =====================
    user_ids = train["userId"].unique()
    movie_ids = train["movieId"].unique()

    user_map = {u: idx for idx, u in enumerate(user_ids)}
    movie_map = {m: idx for idx, m in enumerate(movie_ids)}

    train["u"] = train["userId"].map(user_map)
    train["i"] = train["movieId"].map(movie_map)

    test = test[test["userId"].isin(user_map) & test["movieId"].isin(movie_map)].copy()  # RIVEDI
    test["u"] = test["userId"].map(user_map)
    test["i"] = test["movieId"].map(movie_map)

    n_users = len(user_ids)
    n_items = len(movie_ids)

    # =====================
    # BUILD USER-ITEM MATRIX (sparse)
    # =====================
    R = csr_matrix(
        (train["rating"].astype(float).values, (train["u"].values, train["i"].values)),
        shape=(n_users, n_items)
    )

    # mean-centering per utente (importante)
    user_means = np.array(train.groupby("u")["rating"].mean())
    R_centered = R.copy().astype(float)

    # sottraggo la media utente alle entries non-zero
    rows, cols = R_centered.nonzero()
    R_centered.data = R_centered.data - user_means[rows]

    # NB: full matrix n_users x n_users può essere pesante ma n_users ~ 6000 è gestibile
    S = cosine_similarity(R_centered, dense_output=True)
    np.fill_diagonal(S, 0.0)  # non considero l'utente con se stesso

    # =====================
    # PREDICT FUNCTION
    # =====================
    def predict(u, i, k=50):
        sims = S[u]  # similarità di u con tutti
        # prendo top-k utenti simili
        neigh_idx = np.argpartition(-np.abs(sims), k)[:k]
        neigh_sims = sims[neigh_idx]

        # prendo i rating dei vicini sull'item i (mean-centered)
        neigh_r = R_centered[neigh_idx, i].toarray().ravel()

        # considero solo vicini che hanno rating != 0 su quell'item
        mask = neigh_r != 0
        if not np.any(mask):
            # fallback: media utente
            return float(np.clip(user_means[u], 1, 5))

        num = np.sum(neigh_sims[mask] * neigh_r[mask])
        den = np.sum(np.abs(neigh_sims[mask]))
        if den == 0:
            return float(np.clip(user_means[u], 1, 5))

        pred = user_means[u] + num / den
        return float(np.clip(pred, 1, 5))

    def rmse(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # =====================
    # EVALUATION (sample)
    # =====================
    test_sample = test.sample(n=2000, random_state=42).reset_index(drop=True)

    preds = [predict(int(r.u), int(r.i), k=50) for r in test_sample.itertuples(index=False)]
    score = rmse(test_sample["rating"].values, preds)

    print(f"\nUser-based (sklearn cosine) | k=50 | RMSE (sample 2000) = {score:.4f}")
