import optuna
import pandas as pd
from matrix_factorization import run_mf


def run_optuna(train, val, test, ratings, movies, n_trials=20):

    def objective(trial):
        k = trial.suggest_int("k", 10, 150)
        lr = trial.suggest_float("lr", 5e-4, 5e-2, log=True)
        lambda_reg = trial.suggest_float("lambda_reg", 1e-4, 1e-1, log=True)
        n_epochs = trial.suggest_int("n_epochs", 5, 25)

        val_rmse = run_mf(
            train=train.copy(),
            val=val.copy(),
            test=test.copy(),
            ratings=ratings,
            movies=movies,
            k=k,
            lr=lr,
            lambda_reg=lambda_reg,
            n_epochs=n_epochs,
            verbose=False
        )

        return val_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("\nBest params:", study.best_params)
    print("Best val RMSE:", study.best_value)

    return study
