"""Train the RANDOM FOREST HAB bloom classifier.

500 independent decision trees. Bootstrap sampling gives us Out-of-Bag error
as a built-in validation proxy. For a ~370-row training set, OOB accuracy
typically plateaus by ~500 trees.
"""
from sklearn.ensemble import RandomForestClassifier

from feature_prep import (build_features, evaluate, append_metric_row,
                          save_predictions, save_importance, save_model)

NAME = "RandomForest"
N_TREES = 500

def main():
    features, train, test = build_features()
    X_train, y_train = train[features], train["bloom"]
    X_test,  y_test  = test[features],  test["bloom"]
    print(f"features={len(features)}  train={len(X_train)}  test={len(X_test)}\n")

    model = RandomForestClassifier(
        n_estimators=N_TREES,
        max_depth=None, min_samples_leaf=2,
        class_weight="balanced",
        oob_score=True, bootstrap=True,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"trees built: {N_TREES}")
    print(f"OOB training accuracy (val proxy): {model.oob_score_:.4f}\n")

    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    m = evaluate(NAME, y_test, pred, proba, len(X_train), len(X_test),
                 n_iter=N_TREES)

    save_importance(NAME, features, model.feature_importances_)
    save_predictions(NAME, test, proba, pred)
    save_model(NAME, model)
    append_metric_row(m)
    print(f"\nsaved: models/{NAME}.pkl, data/{NAME}_predictions.csv, "
          f"data/{NAME}_feature_importance.csv")

if __name__ == "__main__":
    main()
