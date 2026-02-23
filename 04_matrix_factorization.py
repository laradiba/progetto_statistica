import pandas as pd
import numpy as np

def run_mf(train, val, test, ratings, movies):
    # Re-indicizzazione (molto importante)
    user_ids = train["userId"].unique()
    movie_ids = train["movieId"].unique()

    user_map = {u: i for i, u in enumerate(user_ids)}
    movie_map = {m: i for i, m in enumerate(movie_ids)}

    train["u"] = train["userId"].map(user_map)
    train["i"] = train["movieId"].map(movie_map)

    test["u"] = test["userId"].map(user_map)
    test["i"] = test["movieId"].map(movie_map)

    train = train.dropna()
    test = test.dropna()

    # =====================
    # PARAMETRI
    # =====================
    n_users = train["u"].nunique()
    n_movies = train["i"].nunique()

    k = 20          # fattori latenti
    lr = 0.01       # learning rate
    lambda_reg = 0.05
    n_epochs = 15

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
        test_rmse = rmse(test)
        print(f"Epoch {epoch:02d} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

    # ===== FUNZIONE DI PREDIZIONE =====
    inv_user_map = {idx: uid for uid, idx in user_map.items()}
    inv_movie_map = {idx: mid for mid, idx in movie_map.items()}

    def predict_rating(userId, movieId):
        # se user o movie non sono nel train → cold start (qui non gestito)
        if userId not in user_map or movieId not in movie_map:
            return None
        u = user_map[userId]
        i = movie_map[movieId]
        pred = mu + b_u[u] + b_i[i] + np.dot(P[u], Q[i])
        return float(np.clip(pred, 1, 5))

    # ===== TOP-N RACCOMANDAZIONI PER UN UTENTE =====
    def recommend_top_n(userId, ratings_df, movies_df, n=10):
        if userId not in user_map:
            return None

        # film già votati dall'utente (nel dataset originale)
        seen = set(ratings_df.loc[ratings_df["userId"] == userId, "movieId"].values)

        # candidati: film presenti nel modello e non ancora visti
        candidates = [mid for mid in movie_map.keys() if mid not in seen]

        # predici
        preds = []
        for mid in candidates:
            pr = predict_rating(userId, mid)
            if pr is not None:
                preds.append((mid, pr))

        preds.sort(key=lambda x: x[1], reverse=True)
        top = preds[:n]

        # aggiungo titolo per interpretazione
        top_df = pd.DataFrame(top, columns=["movieId", "pred_rating"])
        top_df = top_df.merge(movies_df[["movieId", "title", "genres"]], on="movieId", how="left")
        return top_df

    # Esempio: raccomandazioni per un utente a caso (presente nel train)
    example_user = train["userId"].iloc[0]
    top10 = recommend_top_n(example_user, ratings, movies, n=10)
    print("Utente:", example_user)
    print(top10)
