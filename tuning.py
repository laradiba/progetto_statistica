import optuna
import pandas as pd
from matrix_factorization import run_mf

def run_optuna(train, val, test, ratings, movies, n_trials=20):

    # 1) OPTUNA: cerca best params (usa train -> val)
    def objective(trial):
        k = trial.suggest_int("k", 10, 150)
        lr = trial.suggest_float("lr", 5e-4, 5e-2, log=True)
        lambda_reg = trial.suggest_float("lambda_reg", 1e-4, 1e-1, log=True)
        n_epochs = trial.suggest_int("n_epochs", 5, 25)

        # QUI MF viene usato SOLO per valutare val_rmse
        val_rmse = run_mf(
            train=train,
            val=val,
            test=test,          # non usato per scegliere, ma lo passi perché la firma lo richiede
            ratings=ratings,
            movies=movies,
            k=k,
            lr=lr,
            lambda_reg=lambda_reg,
            n_epochs=n_epochs
        )
        return val_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    print("\n===== BEST PARAMS (da validation) =====")
    print(best)
    print("Best VAL RMSE:", study.best_value)

    # 2) FINAL: ri-alleno con best params su train+val, e valuto su test
    print("\n===== FINAL TRAIN (train+val) -> TEST =====")
    train_plus_val = pd.concat([train, val], ignore_index=True)

    test_rmse = run_mf(
        train=train_plus_val,
        val=val,              # qui non conta più per la scelta, puoi passarla comunque
        test=test,
        ratings=ratings,
        movies=movies,
        k=best["k"],
        lr=best["lr"],
        lambda_reg=best["lambda_reg"],
        n_epochs=best["n_epochs"]
    )

    print("FINAL TEST RMSE:", test_rmse)

    return best, test_rmse, study
