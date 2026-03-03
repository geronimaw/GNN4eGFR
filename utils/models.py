from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from interpret.glassbox import ExplainableBoostingClassifier


def get_mlp_classifier():
    return MLPClassifier(
        hidden_layer_sizes=(4,),      # leggermente più grande per equità
        activation='relu',
        solver='adam',
        alpha=1e-3,                   # regolarizzazione L2
        batch_size='auto',
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=42
    )

def get_svm_linear():
    return SVC(
        kernel='linear',
        C=1.0,
        probability=True,
        random_state=42
    )

def get_svm_rbf():
    return SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    )


def get_xgb_classifier(class_counts):
    return XGBClassifier(
        n_estimators=100,        # moderato
        max_depth=3,             # shallow trees
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric='logloss' if len(class_counts) == 2 else 'mlogloss',
        random_state=42
    )

def get_ebm_classifier():
    return ExplainableBoostingClassifier(
        interactions=0,     # solo additive → fairness con KAN shallow
        max_bins=32,
        max_interaction_bins=16,
        learning_rate=0.01,
        random_state=42
    ) # TODO: provare con interactions=10 (più competitivo ma diventa più potente della KAN)

def get_logreg_classifier():
    return LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )