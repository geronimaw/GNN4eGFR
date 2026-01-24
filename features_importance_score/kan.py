import os
import torch
import pandas as pd

fold_path = "/home/alecacciatore/ECML26"
file_path = os.path.join(fold_path, "XY_temp.csv")

# read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# the last column is the target variable and the first one is an index
X = df.iloc[:, 1:-3] # TODO:: include general practitioner features?
y_classif = df.iloc[:, -1]
y_regress = df.iloc[:, -2]

# y_classif contains [I, II, IIIa, IIIb, IV, V] labels
# convert them to numerical labels for classification
label_mapping = {'I': 0, 'II': 1, 'IIIa': 2, 'IIIb': 3, 'IV': 4, 'V': 5}
y_classif = y_classif.map(label_mapping)


# Use Kolmogorov-Arnold networks to classify and regress#
# Score feature importance from both models
from kan import KANClassifier, KANRegressor