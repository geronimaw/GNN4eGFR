import torch
import pandas as pd
from sklearn.model_selection import train_test_split


def get_all_data(file_no_temp_path, task):
    # read the CSV file into a DataFrame
    df = pd.read_csv(file_no_temp_path)

    # remove icd9 columns
    cols_ok = [col for col in df.columns[:-2] if not 'icd9' in col]

    # the last column is the target variable and the first one is an indppex
    y_classif = df.iloc[:, -1]
    y_regress = df.iloc[:, -2]

    # X columns
    # X_no_icd9 = df[cols_ok]
    # print(X_no_icd9.shape, f"with {X_no_icd9.isnull().sum().sum()} missing values")
    X = df.iloc[:, 1:-3]
    print(X.shape, f"with {X.isnull().sum().sum()} missing values")
    X = X.fillna(X.mean())
    print(X.shape, f"with {X.isnull().sum().sum()} missing values")

    # y_classif contains [I, II, IIIa, IIIb, IV, V] labels
    if task == 'binary':
        label_mapping = {'I': 0, 'II': 1, 'IIIa': 1, 'IIIb': 1, 'IV': 1, 'V': 1}
    elif task == '3class':
        label_mapping = {'I': 0, 'II': 1, 'IIIa': 1, 'IIIb': 2, 'IV': 2, 'V': 2}
    y_classif = y_classif.map(label_mapping)

    # Split data into training and testing sets
    X_train, X_test, y_classif_train, y_classif_test = train_test_split(
        X, y_classif, test_size=0.2, random_state=42, stratify=y_classif
    )
    X_train_reg, X_test_reg, y_regress_train, y_regress_test = train_test_split(
        X, y_regress, test_size=0.2, random_state=42
    )

    return (X_train, X_test, X_train_reg, X_test_reg, y_classif, y_classif_train, y_classif_test, y_regress_train, y_regress_test) 