import pandas as pd
import numpy as np

def run_mf(train, val, test, ratings, movies, k, lr, lambda_reg, n_epochs):
    # Re-indicizzazione (molto importante)
    user_ids = train["userId"].unique()
    movie_ids = train["movieId"].unique()

    user_map = {u: i for i, u in enumerate(user_ids)}
    movie_map = {m: i for i, m in enumerate(movie_ids)}

    # Copie per non modificare i df originali
    train = train.copy()
    val = val.copy()
    test = test.copy()

    train["u"] = train["userId"].map(user_map)
    train["i"] = train["movieId"].map(movie_map)

    val["u"] = val["userId"].map(user_map)
    val["i"] = val["movieId"].map(movie_map)

    test["u"] = test["userId"].map(user_map)
    test["i"] = test["movieId"].map(movie_map)

    train = train.dropna()
    val = val.dropna()
    test = test.dropna()

    # =====================
    # PARAMETRI (ora arrivano da input)
    # =====================
    n_users = train["u"].nunique()
    n_movies = train["i"].nunique()

    # =====================
    # INIZIALIZZAZIONE
    # =====================
    mu = train["rating"].mean()

    b_u = np.zeros(n_users)
    b_i = np.zeros(n_movies)

    P = 0.1 * np.random.randn(n_users, k)
    Q = 0.1 * np.random.randn(n_movies, k)

    # =====================
    # TRAINING (SGD)
    # =====================
    def rmse(df):
        preds = mu + b_u[df["u"].astype(int)] + b_i[df["i"].astype(int)] \
                + np.sum(P[df["u"].astype(int)] * Q[df["i"].astype(int)], axis=1)
        return np.sqrt(np.mean((df["rating"].values - preds) ** 2))

    for epoch in range(1, n_epochs + 1):
        for row in train.itertuples():
            u, i, r = int(row.u), int(row.i), row.rating

            pred = mu + b_u[u] + b_i[i] + np.dot(P[u], Q[i])
            err = r - pred

            # update bias
            b_u[u] += lr * (err - lambda_reg * b_u[u])
            b_i[i] += lr * (err - lambda_reg * b_i[i])

            # update fattori latenti
            P[u] += lr * (err * Q[i] - lambda_reg * P[u])
            Q[i] += lr * (err * P[u] - lambda_reg * Q[i])

        train_rmse = rmse(train)
        val_rmse = rmse(val)
        print(f"Epoch {epoch:02d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

    # ✅ Serve a Optuna: restituisco la metrica da minimizzare
    # + restituisco anche il "modello" per poter fare raccomandazioni fuori
    model = {
        "mu": mu,
        "b_u": b_u,
        "b_i": b_i,
        "P": P,
        "Q": Q,
        "user_map": user_map,
        "movie_map": movie_map,
    }
    return float(val_rmse), model


def show_example_recommendations(model, ratings_df, movies_df, n=10):
    mu = model["mu"]
    b_u = model["b_u"]
    b_i = model["b_i"]
    P = model["P"]
    Q = model["Q"]
    user_map = model["user_map"]
    movie_map = model["movie_map"]

    def predict_rating(userId, movieId):
        if userId not in user_map or movieId not in movie_map:
            return None
        u = user_map[userId]
        i = movie_map[movieId]
        pred = mu + b_u[u] + b_i[i] + np.dot(P[u], Q[i])
        return float(np.clip(pred, 1, 5))

    def recommend_top_n(userId, n=10):
        if userId not in user_map:
            return None

        seen = set(ratings_df.loc[ratings_df["userId"] == userId, "movieId"].values)
        candidates = [mid for mid in movie_map.keys() if mid not in seen]

        preds = []
        for mid in candidates:
            pr = predict_rating(userId, mid)
            if pr is not None:
                preds.append((mid, pr))

        preds.sort(key=lambda x: x[1], reverse=True)
        top = preds[:n]

        top_df = pd.DataFrame(top, columns=["movieId", "pred_rating"])
        top_df = top_df.merge(movies_df[["movieId", "title", "genres"]], on="movieId", how="left")
        return top_df

    # scelgo un utente presente nel modello
    rng_ex = np.random.default_rng(123)
    example_user = rng_ex.choice(train["userId"].unique())
   top10 = recommend_top_n(example_user, ratings, movies, n=10)
    print("Utente:", example_user) print(top10)

