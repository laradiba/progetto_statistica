import pandas as pd
import numpy as np

def run_baseline(train, val, test):

    # === METRICA ===
    def rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # =========================
    # BASELINE 1: media globale
    # =========================
    mu = train["rating"].mean()
    pred_mu = np.full(len(test), mu)
    rmse_mu = rmse(test["rating"].values, pred_mu)

    print("\nBaseline 1: mu")
    print("mu =", mu)
    print("RMSE(mu) =", rmse_mu)

    # ===========================================
    # BASELINE 2: mu + bias utente + bias film
    # con regolarizzazione (consigliata)
    # ===========================================
    lambda_reg = 10.0

    # bias film
    movie_stats = train.groupby("movieId")["rating"].agg(["sum", "count"])
    b_i = (movie_stats["sum"] - movie_stats["count"] * mu) / (movie_stats["count"] + lambda_reg)

    # bias utente (dopo aver tolto mu e b_i)
    train_tmp = train.copy()
    train_tmp["b_i"] = train_tmp["movieId"].map(b_i).fillna(0.0)
    train_tmp["residual"] = train_tmp["rating"] - mu - train_tmp["b_i"]

    user_stats = train_tmp.groupby("userId")["residual"].agg(["sum", "count"])
    b_u = user_stats["sum"] / (user_stats["count"] + lambda_reg)

    # predizione test
    test_tmp = test.copy()
    test_tmp["b_i"] = test_tmp["movieId"].map(b_i).fillna(0.0)
    test_tmp["b_u"] = test_tmp["userId"].map(b_u).fillna(0.0)

    pred_bias = (mu + test_tmp["b_u"] + test_tmp["b_i"]).clip(1, 5)
    rmse_bias = rmse(test_tmp["rating"].values, pred_bias.values)

    print("\nBaseline 2: mu + b_u + b_i (reg)")
    print("lambda =", lambda_reg)
    print("RMSE(mu+b_u+b_i) =", rmse_bias)

    # Miglioramento
    improvement = rmse_mu - rmse_bias
    print("\nMiglioramento RMSE:", improvement)

    # Verifica: nessuna coppia (userId, movieId) del test deve stare nel train
    pairs_train = set(zip(train["userId"], train["movieId"]))
    pairs_test  = set(zip(test["userId"], test["movieId"]))
    overlap = len(pairs_train.intersection(pairs_test))
    print("Overlap train-test (user,movie):", overlap)
