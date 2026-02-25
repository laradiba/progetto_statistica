import optuna
import pandas as pd
import numpy as np
from matrix_factorization import run_mf


def run_optuna(train, val, test, ratings, movies, n_trials=20):

    # --- helper: RMSE su test usando il model ritornato da run_mf ---
    def eval_test_rmse(model, test_df):
        mu = model["mu"]
        b_u = model["b_u"]
        b_i = model["b_i"]
        P = model["P"]
        Q = model["Q"]
        user_map = model["user_map"]
        movie_map = model["movie_map"]

        test_df = test_df.copy()
        test_df["u"] = test_df["userId"].map(user_map)
        test_df["i"] = test_df["movieId"].map(movie_map)
        test_df = test_df.dropna()

        u = test_df["u"].astype(int).values
        i = test_df["i"].astype(int).values
        y = test_df["rating"].astype(float).values

        preds = mu + b_u[u] + b_i[i] + np.sum(P[u] * Q[i], axis=1)
        preds = np.clip(preds, 1, 5)

        rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
        return rmse

    # 1) OPTUNA: cerca best params (train -> val)
    def objective(trial):
        k = trial.suggest_int("k", 10, 150)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        lambda_reg = trial.suggest_float("lambda_reg", 1e-3, 1e-1, log=True)
        n_epochs = trial.suggest_int("n_epochs", 5, 10)
        val_rmse, _ = run_mf(
            train=train,
            val=val,
            test=test,
            ratings=ratings,
            movies=movies,
            k=k,
            lr=lr,
            lambda_reg=lambda_reg,
            n_epochs=n_epochs
        )
        if not np.isfinite(val_rmse):
            return float("inf")

        return float(val_rmse)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # se per qualsiasi motivo nessun trial è completato, esci pulito
    if study.best_trial is None:
        print("Nessun trial completato. Probabile instabilità numerica: restringi lr/lambda.")
        return None, None, study

    best = study.best_params
    print("\n===== BEST PARAMS (da validation) =====")
    print(best)
    print("Best VAL RMSE:", study.best_value)

    # 2) FINAL: retrain su train+val, poi valuto su test (una volta sola)
    print("\n===== FINAL TRAIN (train+val) -> TEST =====")
    train_plus_val = pd.concat([train, val], ignore_index=True)

    _, final_model = run_mf(
        train=train_plus_val,
        val=val,
        test=test,
        ratings=ratings,
        movies=movies,
        k=best["k"],
        lr=best["lr"],
        lambda_reg=best["lambda_reg"],
        n_epochs=best["n_epochs"]
    )

    final_test_rmse = eval_test_rmse(final_model, test)
    print("FINAL TEST RMSE:", final_test_rmse)
    from matrix_factorization import show_example_recommendations
    show_example_recommendations(final_model, ratings, movies, n=10)

    return best, final_test_rmse, study
