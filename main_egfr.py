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

fold_path = "/leonardo_work/IscrC_NHPE/ecml26/GNN4eGFR"
out_path = os.path.join(fold_path, "results_eGFR")
file_path = os.path.join(fold_path, "XY_temp.csv")
file_updated_path = os.path.join(fold_path, "XY_temp_updated.csv")
file_no_temp_path = os.path.join(fold_path, "XY_no_temp_updated.csv")

if not os.path.exists(out_path):
    os.makedirs(out_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['binary', '3class'])
    args = parser.parse_args()

    # Get data
    data = get_all_data(file_no_temp_path, args.task)
    X_train, X_test, X_train_reg, X_test_reg, y_classif, y_classif_train, y_classif_test, y_regress_train, y_regress_test = data
    class_counts = y_classif_train.value_counts().sort_index()

    # Create a dataset dictionary for KAN ('test_input', 'test_label', 'train_input', 'train_label')
    train_data_classif = {
        'train_input': torch.tensor(X_train.values, dtype=torch.float32).to(device),
        'train_label': torch.tensor(y_classif_train.values, dtype=torch.long).to(device),
        'test_input': torch.tensor(X_test.values, dtype=torch.float32).to(device),
        'test_label': torch.tensor(y_classif_test.values, dtype=torch.long).to(device)
    }
    train_data_regress = {
        'train_input': torch.tensor(X_train_reg.values, dtype=torch.float32).to(device),
        'train_label': torch.tensor(y_regress_train.values, dtype=torch.float32).to(device),
        'test_input': torch.tensor(X_test_reg.values, dtype=torch.float32).to(device),
        'test_label': torch.tensor(y_regress_test.values, dtype=torch.float32).to(device)
    }

    # Get models
    kan_classifier = KAN(width=[train_data_classif['train_input'].shape[1], 2, 2], grid=3, k=3, device=device, ckpt_path=out_path)
    ebm_classifier = get_ebm_classifier()
    xgb_classifier = get_xgb_classifier(class_counts)
    mlp_classifier = get_mlp_classifier()
    svm_linear = get_svm_linear()
    svm_rbf = get_svm_rbf()
    ebm_classifier = get_ebm_classifier()
    logreg_classifier = get_logreg_classifier()


    runner_classif = ExperimentRunner(class_counts, out_path)

    print("\n\tTraining LogReg")
    runner_classif.run_sklearn_model(
        "LogReg", logreg_classifier,
        X_train, y_classif_train,
        X_test, y_classif_test,
        task="classification"
    )

    # print("\tTraining SVM linear")
    # runner_classif.run_sklearn_model(
    #     "SVM_linear", svm_linear,
    #     X_train, y_classif_train,
    #     X_test, y_classif_test,
    #     task="classification"
    # )

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

    print("\n\tTraining EBM")
    runner_classif.run_sklearn_model(
        "EBM", ebm_classifier,
        X_train, y_classif_train,
        X_test, y_classif_test,
        task="classification"
    )

    print("\n\tTraining MLP")
    runner_classif.run_sklearn_model(
        "MLP", mlp_classifier,
        X_train, y_classif_train,
        X_test, y_classif_test,
        task="classification"
    )

    print("\n\tTraining KAN")
    runner_classif.run_kan_model(
        "KAN 3x3", kan_classifier,
        train_data_classif,
        task="classification",
        steps=20
    )

    print(runner_classif.summary())