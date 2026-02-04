import os
import pandas as pd

xgb_feat_importance_path = "/home/alecacciatore/ECML26/GNN4eGFR/features_importance_score/xgb_scores/xgb_classifier_feature_importances.csv"
kan_feat_importance_path = "/home/alecacciatore/ECML26/GNN4eGFR/features_importance_score/kan_scores/kan_feature_importances.csv"
out_path = "/home/alecacciatore/ECML26/GNN4eGFR/features_importance_score"

# read both csv files
xgb_df = pd.read_csv(xgb_feat_importance_path)
kan_df = pd.read_csv(kan_feat_importance_path)

# replace "Importance" with the position in the dataframe (1 for most important, 2 for second most, etc.)
xgb_df = xgb_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
xgb_df["Importance"] = xgb_df.index + 1

kan_df = kan_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
kan_df["Importance"] = kan_df.index + 1

# compare the feature importance scores
# "Importance" column is irrelevant for comparison since scales differ
comparison_df = pd.merge(xgb_df, kan_df, on="Feature", suffixes=("_XGB", "_KAN"))
comparison_df.to_csv(os.path.join(out_path, "feature_importance_comparison.csv"), index=False)
print("Feature importance comparison saved to feature_importance_comparison.csv")

# compute overall feature importance by averaging the two scores
comparison_df["Importance_Avg"] = (comparison_df["Importance_XGB"] + comparison_df["Importance_KAN"]) / 2
overall_importance_df = comparison_df[["Feature", "Importance_Avg"]].sort_values(by="Importance_Avg", ascending=False)
overall_importance_df.to_csv(os.path.join(out_path, "overall_feature_importance.csv"), index=False)
print("Overall feature importance saved to overall_feature_importance.csv")