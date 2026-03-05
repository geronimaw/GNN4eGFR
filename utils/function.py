import os
import csv
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
    print(y_true.shape, y_proba.shape)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average='binary'),
        "auc": roc_auc_score(y_true, y_proba),
    } if len(class_counts) == 2 else {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average='weighted'),
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
    plt.title(f"ROC - {model_name}" + f"(class #{class_id})" if class_id is not None else
              f"ROC - {model_name}")
    plot_and_save(fig, os.path.join(out_dir, f"{model_name}_roc{'_' + str(class_id)  + 'class' if class_id is not None else ''}.png"))


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
        plt.title(f"Calibration - {model_name}" + f"class #{i}" if len(prob_true) > 2 else
                  f"Calibration - {model_name}")
        plot_and_save(fig, os.path.join(out_dir, f"{model_name}_calibration{'_' + str(i) + 'class' if len(prob_true) > 2 else ''}.png"))
        

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
    

def train_eval_kan(kan_model, train_data, task, class_counts, steps=300, lr=1e-3, output_dir=None):
    optimizer = torch.optim.Adam(kan_model.parameters(), lr=lr)

    X_train = train_data["train_input"]
    y_train = train_data["train_label"]
    X_test = train_data["test_input"]
    y_test = train_data["test_label"]

    train_loss, train_acc, test_metrics = [] , [], []
    best_test_auc = 0.
    
    kan_model.train()

    for step in range(steps):

        optimizer.zero_grad()

        outputs = kan_model(X_train)

        if task == "classification":
            loss = torch.nn.functional.cross_entropy(outputs, y_train)
        else:
            loss = torch.nn.functional.mse_loss(outputs.squeeze(), y_train)

        loss.backward()
        optimizer.step()

        do_test = False
        if (steps > 100 and step % (5 * steps) == 0) or steps < 100:
            train_loss.append(loss.item())

            if task == "classification":
                with torch.no_grad():
                    preds = outputs.argmax(dim=1)
                    acc = (preds == y_train).float().mean().item()

                train_acc.append(acc)

            print(f"iter {step}/{steps}\n\t\ttrain loss = {train_loss[-1]}\ttrain_acc = {acc}")
            do_test = True
    # kan_model.fit(train_data, steps=steps,
    #               loss_fn=torch.nn.CrossEntropyLoss(), lr=lr)

    if do_test:
        kan_model.eval()

        with torch.no_grad():
            outputs = kan_model(X_test)

            if task == "classification":
                probs = torch.softmax(outputs, dim=1)[:, 1] if len(class_counts) == 2 else torch.softmax(outputs, dim=1)
                y_pred = torch.argmax(outputs, dim=1)
                y_pred_np = y_pred.cpu().numpy()
                y_proba_np = probs.cpu().numpy()
                y_test_np = y_test.cpu().numpy()

                test_metrics.append(compute_classification_metrics(y_test_np, y_pred_np, y_proba_np, class_counts))
            else:
                y_pred = outputs.squeeze()
                y_pred_np = y_pred.cpu().numpy()
                y_test_np = y_test.cpu().numpy()
                test_metrics.append(compute_regression_metrics(y_test_np, y_pred_np))
       
        if test_metrics[-1]["accuracy"] > best_test_auc:
            best_test_auc = test_metrics[-1]["accuracy"]

            y_test = train_data["test_label"].cpu().numpy()
            if task == "classification":
                if len(class_counts) == 2:
                    plot_roc(y_test, y_proba_np, "KAN (best test AUC)", output_dir)
                else:
                    from sklearn.preprocessing import label_binarize
                    y_test_bin = label_binarize(y_test, classes=range(len(class_counts)))
                    # y_proba = kan_model(train_data["test_input"])
                    fpr = dict()
                    tpr = dict()

                    for i in range(len(class_counts)):
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba_np[:, i])
                        plot_roc(fpr[i], tpr[i], "KAN (best test AUC)", output_dir, i)
                plot_calibration(y_test, y_proba_np, "KAN (best test AUC)", output_dir, class_counts)

                print(f"\t\ttest auc (new best) = {best_test_auc}")
                best_test_metrics = test_metrics[-1]
        
        else:
            print(f"\t\tbest test auc still = {best_test_auc}")
     

    return train_loss, train_acc, test_metrics, best_test_metrics  


class ExperimentRunner:
    def __init__(self, class_counts, output_dir="results"):
        self.output_dir = output_dir
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
                # y_proba = model.predict_proba(X_test)
                fpr = dict()
                tpr = dict()

                for i in range(len(self.class_counts)):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    plot_roc(fpr[i], tpr[i], name, self.output_dir, i)
            plot_calibration(y_test, y_proba, name, self.output_dir, self.class_counts)

        self.results[name] = metrics

    def run_kan_model(self, name, kan_model, train_data, task, steps):
        train_loss, train_acc, test_metrics, best_test_metrics = train_eval_kan(
            kan_model, train_data, task, self.class_counts, steps=steps, output_dir=self.output_dir
        )
            # plot_roc(y_test, y_proba, name, self.output_dir)
            # plot_calibration(y_test, y_proba, name, self.output_dir)
        
        history = []
        for idx, iter_metric in enumerate(test_metrics):
            iter_metric["step"] = idx//100*steps
            iter_metric["train_loss"] = train_loss[idx] 
            iter_metric["train_acc"] = train_acc[idx] 
            history.append(iter_metric)
        
        fieldnames = iter_metric.keys()
        with open(os.path.join(self.output_dir, "history.csv"), "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

        self.results[name] = best_test_metrics

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
