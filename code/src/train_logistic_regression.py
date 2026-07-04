"""Train the LOGISTIC REGRESSION HAB bloom classifier.

Convex objective; L-BFGS converges in tens of iterations for this size, so
max_iter=5000 is just a safety ceiling. `n_iter_` on the fitted model tells
us the actual work done.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from feature_prep import (build_features, evaluate, append_metric_row,
                          save_predictions, save_importance, save_model)

NAME = "LogisticRegression"
MAX_ITER = 5000

def main():
    features, train, test = build_features()
    X_train, y_train = train[features], train["bloom"]
    X_test,  y_test  = test[features],  test["bloom"]
    print(f"features={len(features)}  train={len(X_train)}  test={len(X_test)}\n")

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf",   LogisticRegression(
            max_iter=MAX_ITER, solver="lbfgs", C=1.0,
            class_weight="balanced", random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    actual_iter = int(model.named_steps["clf"].n_iter_[0])
    print(f"actual iterations to converge: {actual_iter} / {MAX_ITER}\n")

    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    m = evaluate(NAME, y_test, pred, proba, len(X_train), len(X_test),
                 n_iter=actual_iter)

    coefs = model.named_steps["clf"].coef_.ravel()
    save_importance(NAME, features, np.abs(coefs), signed=coefs)
    save_predictions(NAME, test, proba, pred)
    save_model(NAME, model)
    append_metric_row(m)
    print(f"\nsaved: models/{NAME}.pkl, data/{NAME}_predictions.csv, "
          f"data/{NAME}_feature_importance.csv")

if __name__ == "__main__":
    main()
