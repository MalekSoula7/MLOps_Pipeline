import mlflow, mlflow.sklearn, pandas as pd, joblib, json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("heart-disease-classifier")

df = pd.read_csv("data/raw/heart.csv")
X, y = df.drop("target", axis=1), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {"n_estimators": 100, "max_depth": 8, "min_samples_split": 5}

with mlflow.start_run(run_name="rf-baseline") as run:
    mlflow.log_params(params)
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", RandomForestClassifier(**params, random_state=42))])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    metrics = {"auc": round(roc_auc_score(y_test, y_prob), 4),
               "f1":  round(f1_score(y_test, y_pred), 4)}
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(pipe, "model",
        registered_model_name="heart-disease-classifier")

    joblib.dump(pipe, "model.pkl")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)
    print(f"AUC: {metrics['auc']}  F1: {metrics['f1']}")