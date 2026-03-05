import os
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from kan import KAN
from utils.get_data import get_all_data
from utils.function import ExperimentRunner
from sklearn.preprocessing import StandardScaler


fold_path = "./"
out_path = os.path.join(fold_path, "results_eGFR")

data_files = {
    "T": os.path.join(fold_path, "XY_temp.csv"),
    "Tupd": os.path.join(fold_path, "XY_temp_updated.csv"),
    "noT": os.path.join(fold_path, "XY_no_temp_updated.csv")
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument('--task', choices=['binary', '3class'], default='binary')
    parser.add_argument('--data', choices=['T', 'Tupd', 'noT'], default='noT')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--lamb', type=float, default=0.)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--grid', type=int, default=6)
    parser.add_argument('--samePrep4all', type=bool, default=False)
    args = parser.parse_args()
    
    vars = args.file_name.split("_")
    internal_args = [
        "--file_name", args.file_name,
        "--task", vars[0],
        "--data", vars[1],
        "--steps", vars[2],
        "--hidden", vars[3],
        "--grid", vars[4],
        "--lamb", vars[5]
        ]

    args = parser.parse_args(internal_args)

    out_path = os.path.join(out_path, "KAN_optim2", args.file_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Get data
    data = get_all_data(data_files[args.data], args.task)
    X_train, X_test, X_train_reg, X_test_reg, y_classif, y_classif_train, y_classif_test, y_regress_train, y_regress_test = data
    class_counts = y_classif_train.value_counts().sort_index()

    # Create a dataset dictionary for KAN ('test_input', 'test_label', 'train_input', 'train_label')

    if args.samePrep4all:
        X_train = X_train.values
        X_test = X_test.values
        X_train_reg = X_train_reg.values
        X_test_reg = X_test_reg.values
    else:
        scaler_classif = StandardScaler()
        scaler_regress = StandardScaler()

        X_train = scaler_classif.fit_transform(X_train)
        X_test = scaler_classif.transform(X_test)
        X_train_reg = scaler_regress.fit_transform(X_train_reg)
        X_test_reg = scaler_regress.transform(X_test_reg)

    train_data_classif = {
        'train_input': torch.tensor(X_train, dtype=torch.float32).to(device),
        'train_label': torch.tensor(y_classif_train.values, dtype=torch.long).to(device),
        'test_input': torch.tensor(X_test, dtype=torch.float32).to(device),
        'test_label': torch.tensor(y_classif_test.values, dtype=torch.long).to(device)
    }
    train_data_regress = {
        'train_input': torch.tensor(X_train_reg, dtype=torch.float32).to(device),
        'train_label': torch.tensor(y_regress_train.values, dtype=torch.float32).to(device),
        'test_input': torch.tensor(X_test_reg, dtype=torch.float32).to(device),
        'test_label': torch.tensor(y_regress_test.values, dtype=torch.float32).to(device)
    }
    
    csv_path = os.path.join(out_path, "kan_hyperparameter_search.csv")

    if os.path.exists(csv_path):
        print("Loading previous hyperparameter search...")
        df_old = pd.read_csv(csv_path)
    else:
        df_old = pd.DataFrame(columns=["hidden","grid","k","accuracy","f1","auc"])

    # Get models
    width = [train_data_classif['train_input'].shape[1], args.hidden, 2 if args.task == 'binary' else 3]
    print(f"\tTraining KAN with shape {width} and grid={args.grid}")

    kan_classifier = KAN(
            width=width,
            grid=args.grid,
            k=3,
            device=device,
            ckpt_path=out_path
    )
    runner_classif = ExperimentRunner(class_counts, out_path)

    runner_classif.run_kan_model(
        "KAN",
        kan_classifier,
        train_data_classif,
        task="classification",
        steps=args.steps,
        # lamb=float(args.lamb) # TODO: IMPORTANTE addestrare con kan.fit altrimenti no regressione simbolica
    )

    summary = runner_classif.summary()

    print(summary)

    results = {
        "hidden": args.hidden,
        "grid": args.grid,
        "k": 3,
        "accuracy": summary.loc["KAN", "accuracy"],
        "f1": summary.loc["KAN", "f1"],
        "auc": summary.loc["KAN", "auc"]
    }

    # df_results = pd.DataFrame(results)

    # df_results.to_csv(os.path.join(out_path, f"kan_hyperparameter_search.csv"), index=False)
    df_results = pd.DataFrame([results])

    df_results.to_csv(
        os.path.join(out_path, "kan_hyperparameter_search.csv"),
        index=False
    )