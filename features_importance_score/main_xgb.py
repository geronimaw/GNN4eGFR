import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


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


##### XGBoost Classifier #####
# XGBoost Classifier will be used for feature importance extraction
classifier = xgb.XGBClassifier(objective='binary:logistic', missing=-999, importance_type='weight')

parameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
}

model_classif = GridSearchCV(estimator=classifier,
                             param_grid=parameters,
                             scoring='accuracy',
                             n_jobs=-1,
                             cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                             verbose=1)
model_classif.fit(X, y_classif)

# save best parameters model
best_params_classif = model_classif.best_params_
with open(os.path.join(fold_path, "xgb_classifier_best_params.txt"), "w") as f:
    for param, value in best_params_classif.items():
        f.write(f"{param}: {value}\n")

# fit the model with best parameters on the entire dataset
best_model = model_classif.best_estimator_
best_model.fit(X, y_classif)
# save feature importances
feature_importances = best_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df.to_csv(os.path.join(fold_path, "xgb_feature_importances.csv"), index=False)

##### XGBoost Regressor #####
regressor = xgb.XGBRegressor(objective='reg:squarederror', missing=-999, importance_type='weight')
parameters_reg = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
}

model_regress = GridSearchCV(estimator=regressor,
                             param_grid=parameters_reg,
                             scoring='neg_mean_squared_error',
                             n_jobs=-1,
                             cv=5,
                             verbose=1)
model_regress.fit(X, y_regress)
# save best parameters model
best_params_regress = model_regress.best_params_
with open(os.path.join(fold_path, "xgb_regressor_best_params.txt"), "w") as f:
    for param, value in best_params_regress.items():
        f.write(f"{param}: {value}\n")

# fit the model with best parameters on the entire dataset
best_model_reg = model_regress.best_estimator_
best_model_reg.fit(X, y_regress)
# save feature importances
feature_importances_reg = best_model_reg.feature_importances_
importance_df_reg = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances_reg})
importance_df_reg = importance_df_reg.sort_values(by='Importance', ascending=False)
importance_df_reg.to_csv(os.path.join(fold_path, "xgb_regressor_feature_importances.csv"), index=False)
