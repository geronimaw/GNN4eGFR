import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, roc_curve,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.model_selection import StratifiedKFold, KFold


def compute_classification_metrics(y_true, y_pred, y_proba, class_counts):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average='binary' if len(class_counts) == 2 else 'weighted'),
        "auc": roc_auc_score(y_true, y_proba, multi_class='ovr'),
    }


def compute_regression_metrics(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }


def bootstrap_ci(metric_fn, y_true, y_pred, n_bootstrap=1000, alpha=0.95):
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    lower = np.percentile(scores, (1-alpha)/2*100)
    upper = np.percentile(scores, (1+alpha)/2*100)
    return lower, upper

def plot_and_save(fig, save_path):
    fig.savefig(save_path)
    plt.close(fig)


def plot_roc(y_true, y_proba, model_name, out_dir, class_id=None):
    if class_id is None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
    else:
        fpr, tpr = y_true, y_proba
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC - {model_name} (class #{class_id})")
    plot_and_save(fig, os.path.join(out_dir, f"{model_name}_roc_{class_id}class.png"))


def plot_calibration(y_true, y_proba, model_name, out_dir, class_counts):
    if len(class_counts) == 2:
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        prob_true = [prob_true]
        prob_pred = [prob_pred]
    else:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_true, classes=range(len(class_counts)))
        
        prob_true = dict()
        prob_pred = dict()

        for i in range(len(class_counts)):
            prob_true[i], prob_pred[i] = calibration_curve(
                y_test_bin[:, i],
                y_proba[:, i],
                n_bins=10
            )
    
    for i in range(len(prob_true)):
        fig = plt.figure()
        plt.plot(prob_pred[i], prob_true[i])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Calibration - {model_name}" + f"class #{i}" if len(prob_true) > 2 else '')
        plot_and_save(fig, os.path.join(out_dir, f"{model_name}_calibration" +
                                         f"_{i}class" if len(prob_true) > 2 else '' + ".png"))
        

def train_eval_sklearn(model, X_train, y_train, X_test, y_test, task, class_counts):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task == "classification":
        y_proba = model.predict_proba(X_test)
        probs = y_proba[:, 1] if len(class_counts) == 2 else y_proba
        metrics = compute_classification_metrics(y_test, y_pred, probs, class_counts)
        return metrics, y_pred, probs
    else:
        metrics = compute_regression_metrics(y_test, y_pred)
        return metrics, y_pred, None
    

def train_eval_kan(kan_model, train_data, task, class_counts, steps=300, lr=1e-3):
    optimizer = torch.optim.Adam(kan_model.parameters(), lr=lr)

    X_train = train_data["train_input"]
    y_train = train_data["train_label"]
    X_test = train_data["test_input"]
    y_test = train_data["test_label"]

    kan_model.train()

    for _ in range(steps):
        optimizer.zero_grad()
        outputs = kan_model(X_train)

        if task == "classification":
            loss = torch.nn.functional.cross_entropy(outputs, y_train)
        else:
            loss = torch.nn.functional.mse_loss(outputs.squeeze(), y_train)

        loss.backward()
        optimizer.step()

    kan_model.eval()
    with torch.no_grad():
        outputs = kan_model(X_test)

        if task == "classification":
            probs = torch.softmax(outputs, dim=1)[:, 1]
            y_pred = torch.argmax(outputs, dim=1)
            y_pred_np = y_pred.cpu().numpy()
            y_proba_np = probs.cpu().numpy()
            y_test_np = y_test.cpu().numpy()

            metrics = compute_classification_metrics(
                y_test_np, y_pred_np, y_proba_np, class_counts
            )
            return metrics, y_pred_np, y_proba_np
        else:
            y_pred = outputs.squeeze()
            y_pred_np = y_pred.cpu().numpy()
            y_test_np = y_test.cpu().numpy()
            metrics = compute_regression_metrics(y_test_np, y_pred_np)
            return metrics, y_pred_np, None
        

class ExperimentRunner:
    def __init__(self, class_counts, output_dir="results"):
        self.output_dir = output_dir + "_binary" if len(class_counts) == 2 else output_dir + f"_{len(class_counts)}class"
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = {}
        self.class_counts = class_counts

    def run_sklearn_model(self, name, model, X_train, y_train, X_test, y_test, task):
        metrics, y_pred, y_proba = train_eval_sklearn(
            model, X_train, y_train, X_test, y_test, task, self.class_counts
        )

        if task == "classification":
            if len(self.class_counts) == 2:
                plot_roc(y_test, y_proba, name, self.output_dir)
            else:
                from sklearn.preprocessing import label_binarize
                y_test_bin = label_binarize(y_test, classes=range(len(self.class_counts)))
                y_proba = model.predict_proba(X_test)
                fpr = dict()
                tpr = dict()

                for i in range(len(self.class_counts)):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    plot_roc(fpr[i], tpr[i], name, self.output_dir, i)
            plot_calibration(y_test, y_proba, name, self.output_dir, self.class_counts)

        self.results[name] = metrics

    def run_kan_model(self, name, kan_model, train_data, task, steps):
        metrics, y_pred, y_proba = train_eval_kan(
            kan_model, train_data, task, self.class_counts, steps
        )

        y_test = train_data["test_label"].cpu().numpy()

        if task == "classification":
            plot_roc(y_test, y_proba, name, self.output_dir)
            plot_calibration(y_test, y_proba, name, self.output_dir)

        self.results[name] = metrics

    def summary(self):
        df = pd.DataFrame(self.results).T
        df.to_csv(os.path.join(self.output_dir, "summary.csv"))
        return df
    

def cross_validate(model_fn, X, y, task, n_splits=5):
    if task == "classification":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        metrics, _, _ = model_fn(X_tr, y_tr, X_te, y_te)
        results.append(metrics)

    return pd.DataFrame(results).mean(), pd.DataFrame(results).std()
