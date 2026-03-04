import os
import torch
import argparse

from kan import KAN
from utils.get_data import get_all_data
from utils.models import (
    get_ebm_classifier, get_mlp_classifier, get_svm_linear,
    get_svm_rbf, get_xgb_classifier, get_mlp_classifier, get_logreg_classifier
)
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
    args = parser.parse_args()

    out_path = os.path.join(out_path, f"{args.data}_{args.task}")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Get data
    data = get_all_data(data_files[args.data], args.task)
    X_train, X_test, X_train_reg, X_test_reg, y_classif, y_classif_train, y_classif_test, y_regress_train, y_regress_test = data
    class_counts = y_classif_train.value_counts().sort_index()

    # Create a dataset dictionary for KAN ('test_input', 'test_label', 'train_input', 'train_label')
    scaler_classif = StandardScaler()
    scaler_regress = StandardScaler()

    # Classification
    X_train_scaled = scaler_classif.fit_transform(X_train)
    X_test_scaled = scaler_classif.transform(X_test)

    # Regression
    X_train_reg_scaled = scaler_regress.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_regress.transform(X_test_reg)

    train_data_classif = {
        'train_input': torch.tensor(X_train_scaled, dtype=torch.float32).to(device),
        'train_label': torch.tensor(y_classif_train.values, dtype=torch.long).to(device),
        'test_input': torch.tensor(X_test_scaled, dtype=torch.float32).to(device),
        'test_label': torch.tensor(y_classif_test.values, dtype=torch.long).to(device)
    }
    train_data_regress = {
        'train_input': torch.tensor(X_train_reg_scaled, dtype=torch.float32).to(device),
        'train_label': torch.tensor(y_regress_train.values, dtype=torch.float32).to(device),
        'test_input': torch.tensor(X_train_reg_scaled, dtype=torch.float32).to(device),
        'test_label': torch.tensor(y_regress_test.values, dtype=torch.float32).to(device)
    }

    # Get models
    kan_classifier = KAN(width=[train_data_classif['train_input'].shape[1], 16, 2 if args.task == 'binary' else 3], 
                         grid=3, k=3, device=device, ckpt_path=out_path)
    ebm_classifier = get_ebm_classifier()
    xgb_classifier = get_xgb_classifier(class_counts)
    mlp_classifier = get_mlp_classifier()
    svm_linear = get_svm_linear()
    svm_rbf = get_svm_rbf()
    logreg_classifier = get_logreg_classifier()

    # Train the models
    runner_classif = ExperimentRunner(class_counts, out_path)

    # print("\n\tTraining LogReg")
    # runner_classif.run_sklearn_model(
    #     "LogReg", logreg_classifier,
    #     X_train, y_classif_train,
    #     X_test, y_classif_test,
    #     task="classification"
    # )

    # # print("\tTraining SVM linear")
    # # runner_classif.run_sklearn_model(
    # #     "SVM_linear", svm_linear,
    # #     X_train, y_classif_train,
    # #     X_test, y_classif_test,
    # #     task="classification"
    # # )

    # print("\tTraining SVM rbf")
    # runner_classif.run_sklearn_model(
    #     "SVM_rbf", svm_rbf,
    #     X_train, y_classif_train,
    #     X_test, y_classif_test,
    #     task="classification"
    # )

    print("\n\tTraining XGB")
    runner_classif.run_sklearn_model(
        "XGBoost", xgb_classifier,
        X_train, y_classif_train,
        X_test, y_classif_test,
        task="classification"
    )

    # print("\n\tTraining EBM")
    # runner_classif.run_sklearn_model(
    #     "EBM", ebm_classifier,
    #     X_train, y_classif_train,
    #     X_test, y_classif_test,
    #     task="classification"
    # )

    # print("\n\tTraining MLP")
    # runner_classif.run_sklearn_model(
    #     "MLP", mlp_classifier,
    #     X_train, y_classif_train,
    #     X_test, y_classif_test,
    #     task="classification"
    # )

    print("\n\tTraining KAN")
    runner_classif.run_kan_model(
        "KAN", kan_classifier,
        train_data_classif,
        task="classification",
        steps=20
    )

    print(runner_classif.summary())
