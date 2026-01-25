import os
import torch
import numpy as np
import pandas as pd
from kan import KAN, ex_round

fold_path = "/home/alecacciatore/ECML26/GNN4eGFR"
out_path = os.path.join(fold_path, "features_importance_score/kan_scores")
file_path = os.path.join(fold_path, "XY_temp.csv")

if not os.path.exists(out_path):
    os.makedirs(out_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# the last column is the target variable and the first one is an indppex
X = df.iloc[:, 1:-3] # TODO:: include general practitioner features?
y_classif = df.iloc[:, -1]
y_regress = df.iloc[:, -2]

# y_classif contains [I, II, IIIa, IIIb, IV, V] labels
# convert them to numerical labels for classification
# label_mapping = {'I': 0, 'II': 1, 'IIIa': 2, 'IIIb': 3, 'IV': 4, 'V': 5}
label_mapping = {'I': 0, 'II': 1, 'IIIa': 1, 'IIIb': 1, 'IV': 1, 'V': 1}
y_classif = y_classif.map(label_mapping)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_classif_train, y_classif_test = train_test_split(
    X, y_classif, test_size=0.2, random_state=42, stratify=y_classif
)
X_train_reg, X_test_reg, y_regress_train, y_regress_test = train_test_split(
    X, y_regress, test_size=0.2, random_state=42
)

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

##### Use Kolmogorov-Arnold networks to classify and regress#
# Score feature importance from both models
# KAN Classifier
print("Training KAN Classifier...")
kan_classifier = KAN(width=[train_data_classif['train_input'].shape[1], 5, 2], grid=3, k=3, device=device)
# kan_classifier.speed()

def train_acc():
    return torch.mean((torch.argmax(kan_classifier(train_data_classif['train_input']), dim=1) == train_data_classif['train_label']).type(torch.float32))

def test_acc():
    return torch.mean((torch.argmax(kan_classifier(train_data_classif['test_input']), dim=1) == train_data_classif['test_label']).type(torch.float32))

results = kan_classifier.fit(train_data_classif, opt="LBFGS", steps=20, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss())
print("Training completed."
      f"\nFinal Train Accuracy: {results['train_acc'][-1]:.4f}"
      f"\nFinal Test Accuracy: {results['test_acc'][-1]:.4f}")

# prune the network
kan_classifier.prune(threshold=0.01)

# test the pruned network
print("Testing pruned KAN Classifier...")
print("Train Accuracy after pruning:", train_acc().item())
print("Test Accuracy after pruning:", test_acc().item())

# get feature importance scores
importance_scores = {}
for layer in kan_classifier.layers:
    importance_scores[layer] = layer.coef



# # Retrieve symbolic regression
# lib = ['x','x^2','x^3','exp','log','sqrt']
# kan_classifier.auto_symbolic(lib=lib)
# formula = kan_classifier.symbolic_formula()[0][0]
# ex_round(formula, 4)

# def acc(formula, X, y):
#     batch = X.shape[0]
#     correct = 0
#     for i in range(batch):
#         correct += np.round(np.array(formula.subs('x_1', X[i,0]).subs('x_2', X[i,1])).astype(np.float64)) == y[i,0]
#     return correct/batch

# print('train acc of the formula:', acc(formula, train_data_classif['train_input'], train_data_classif['train_label']))
# print('test acc of the formula:', acc(formula, train_data_classif['test_input'], train_data_classif['test_label']))

# # order features by importance
# feature_importances_classif = kan_classifier.feature_importance(train_data_classif['train_input'], train_data_classif['train_label'])
# importance_df_classif = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances_classif})
# importance_df_classif = importance_df_classif.sort_values(by='Importance', ascending=False)
# print(importance_df_classif)

# # save to CSV to os.path.join(out_path, "kan_classifier_feature_importances.csv")
# importance_df_classif.to_csv(os.path.join(out_path, "kan_classifier_feature_importances.csv"), index=False)

# # KAN Regressor
# kan_regressor = KAN(width=[2,1], grid=3, k=3, device=device)
# kan_regressor.speed()
# kan_regressor.train(train_data_regress['train_input'], train_data_regress['train_label'], epochs=100, batch_size=32)
# feature_importances_regress = kan_regressor.feature_importance(train_data_regress['train_input'], train_data_regress['train_label'])
# importance_df_regress = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances_regress})
# importance_df_regress = importance_df_regress.sort_values(by='Importance', ascending=False)
# importance_df_regress.to_csv(os.path.join(out_path, "kan_regressor_feature_importances.csv"), index=False)

