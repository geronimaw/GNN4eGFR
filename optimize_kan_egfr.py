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
    "temp": os.path.join(fold_path, "XY_temp.csv"),
    "temp_updated": os.path.join(fold_path, "XY_temp_updated.csv"),
    "no_temp": os.path.join(fold_path, "XY_no_temp_updated.csv")
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['binary', '3class'])
    parser.add_argument('--data', choices=['temp', 'temp_updated', 'no_temp'])
    parser.add_argument('--samePrep4all', type=bool, default=False)
    args = parser.parse_args()

    out_path = os.path.join(out_path, "KAN_optim_prova", f"{args.task}_{args.data}")
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

    results = []

    # Get models
    # for hidden in [2, 4, 8, 16, 32, 40, 64]:
    for hidden in [70, 80, 100, 128]:
        for grid in [3, 4, 5]:
            width = [train_data_classif['train_input'].shape[1], hidden, 2 if args.task == 'binary' else 3]
            print(f"\tTraining KAN with shape {width} and grid={grid}")

            out_kan_path = os.path.join(out_path, f"h{hidden}g{grid}")
            if not os.path.exists(out_kan_path):
                os.makedirs(out_kan_path)

            kan_classifier = KAN(
                    width=width,
                    grid=grid,
                    k=3,
                    device=device,
                    ckpt_path=out_kan_path
            )
            runner_classif = ExperimentRunner(class_counts, out_kan_path)

            runner_classif.run_kan_model(
                "KAN",
                kan_classifier,
                train_data_classif,
                task="classification",
                steps=20
            )

            summary = runner_classif.summary()

            results.append({
                "hidden": hidden,
                "grid": grid,
                "k": 3,
                "accuracy": summary.loc["KAN", "accuracy"],
                "f1": summary.loc["KAN", "f1"],
                "auc": summary.loc["KAN", "auc"]
            })

    # df_results = pd.DataFrame(results)
    df_new = pd.DataFrame(results)
    df_results = pd.concat([df_old, df_new], ignore_index=True)
    print(df_results)

    df_results.to_csv(os.path.join(out_path, f"kan_hyperparameter_search.csv"), index=False)
    best_f1 = df_results.loc[df_results["f1"].idxmax()]
    print("Best F1 configuration:", best_f1)
    best_auc = df_results.loc[df_results["auc"].idxmax()]
    print("Best AUC configuration:", best_auc)

    with open(os.path.join(out_path, f"best_configs.csv"), "w") as text_file:
        text_file.write(f"Best F1 configuration: \n{best_f1}")
        text_file.write(f"\n\nBest AUC configuration: \n{best_auc}")
    
    ##### plots
    # F1 score
    for grid in df_results.grid.unique():
        subset = df_results[df_results.grid == grid]
        plt.plot(subset.hidden, subset.f1, marker="o", label=f"grid={grid}")

    plt.xlabel("Hidden units")
    plt.ylabel("F1 score")
    plt.legend()
    plt.title("KAN performance vs hidden size")
    plt.savefig(os.path.join(out_path, f"F1_vs_hidden.png"))
    plt.close()

    pivot = df_results.pivot_table(values="f1", index="hidden", columns="grid", aggfunc="mean")

    sns.heatmap(pivot, annot=True, cmap="viridis")
    plt.title("F1")
    plt.savefig(os.path.join(out_path, "F1_grid_vs_hidden.png"))
    plt.close()
    
    # AUC
    for grid in df_results.grid.unique():
        subset = df_results[df_results.grid == grid]
        plt.plot(subset.hidden, subset.auc, marker="o", label=f"grid={grid}")

    plt.xlabel("Hidden units")
    plt.ylabel("AUC")
    plt.legend()
    plt.title("KAN performance vs hidden size")
    plt.savefig(os.path.join(out_path, f"AUC_vs_hidden.png"))

    pivot = df_results.pivot_table(
        values="auc",
        index="hidden",
        columns="grid",
        aggfunc="mean"
    )

    sns.heatmap(pivot, annot=True, cmap="viridis")

    plt.title("AUC")
    plt.savefig(os.path.join(out_path, "AUC_grid_vs_hidden.png"))
    plt.close()
