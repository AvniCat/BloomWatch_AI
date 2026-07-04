# Results

Outputs from the training scripts. All files are auto-generated — running
`code/src/train_*.py` recreates them.

## Structure

```
results/
├── models/                     pickled scikit-learn / xgboost models
│   ├── LogisticRegression.pkl
│   ├── RandomForest.pkl
│   └── XGBoost.pkl
├── predictions/                per-model test-set predictions
│   ├── LogisticRegression_predictions.csv
│   ├── RandomForest_predictions.csv
│   └── XGBoost_predictions.csv
├── feature_importance/         per-model feature ranking
│   ├── LogisticRegression_feature_importance.csv
│   ├── RandomForest_feature_importance.csv
│   └── XGBoost_feature_importance.csv
├── figures/                    (add plots / screenshots / diagrams here)
└── model_metrics.csv           one row per model: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
```

## Loading a trained model

```python
import pickle
with open("results/models/RandomForest.pkl", "rb") as f:
    rf = pickle.load(f)
proba = rf.predict_proba(X_new)[:, 1]   # X_new must have 25 features in the same order
```
