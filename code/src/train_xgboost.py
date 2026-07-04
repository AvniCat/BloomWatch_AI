"""Train the XGBOOST HAB bloom classifier.

Up to N_ROUNDS_MAX boosting rounds with early stopping. The last 20% of the
training set (chronologically) is held out as a validation split; boosting
stops when val log-loss hasn't improved for `early_stopping_rounds` rounds.
"""
from xgboost import XGBClassifier

from feature_prep import (build_features, evaluate, append_metric_row,
                          save_predictions, save_importance, save_model)

NAME = "XGBoost"
N_ROUNDS_MAX = 2000
EARLY_STOP   = 50

def main():
    features, train, test = build_features()
    X_train, y_train = train[features], train["bloom"]
    X_test,  y_test  = test[features],  test["bloom"]
    print(f"features={len(features)}  train={len(X_train)}  test={len(X_test)}\n")

    # temporal val split within the training window
    train_sorted = train.sort_values(["year", "month", "region"]).reset_index(drop=True)
    cut = int(len(train_sorted) * 0.80)
    Xtr = train_sorted.iloc[:cut][features]
    ytr = train_sorted.iloc[:cut]["bloom"]
    Xva = train_sorted.iloc[cut:][features]
    yva = train_sorted.iloc[cut:]["bloom"]
    scale_pos_w = (ytr == 0).sum() / max(1, (ytr == 1).sum())

    model = XGBClassifier(
        n_estimators=N_ROUNDS_MAX,
        max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_w,
        eval_metric="logloss",
        early_stopping_rounds=EARLY_STOP,
        random_state=42, n_jobs=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    best = int(model.best_iteration)
    print(f"early-stopped at round: {best} / {N_ROUNDS_MAX}  "
          f"(val log-loss = {model.best_score:.4f})\n")

    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    m = evaluate(NAME, y_test, pred, proba, len(X_train), len(X_test), n_iter=best)

    save_importance(NAME, features, model.feature_importances_)
    save_predictions(NAME, test, proba, pred)
    save_model(NAME, model)
    append_metric_row(m)
    print(f"\nsaved: models/{NAME}.pkl, data/{NAME}_predictions.csv, "
          f"data/{NAME}_feature_importance.csv")

if __name__ == "__main__":
    main()
