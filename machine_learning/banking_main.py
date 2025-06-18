from __future__ import annotations

from dgh_processing.anjana.banking_dgh import get_csv_banking_dghs

from pathlib import Path
import csv

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessor import Preprocessor, encode_features

try:
	from xgboost import XGBClassifier
except ImportError:
	XGBClassifier = None

TARGET_COL = "y"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

STRATEGY = {
	"age": "interval",
	"balance": "interval",
	"education": "one_hot",
	"job": "one_hot",
	"marital": "one_hot",
	"month": "one_hot",
	"previous": "interval",
}

# ---------------------------------------------------------------------------
# Model registry with grid definitions
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
	"logistic": (LogisticRegression(max_iter=100000, n_jobs=-1), {"C": [0.1, 1, 10]}),
	"tree": (
		DecisionTreeClassifier(random_state=RANDOM_STATE),
		{"max_depth": [5, 10, None], "min_samples_split": [2, 5]},
	),
	"forest": (
		RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
		{
			"n_estimators": [100, 200],
			"max_depth": [5, 10, None],
			"min_samples_split": [2, 5],
		},
	),
	"mlp": (
		MLPClassifier(
			hidden_layer_sizes=(16, 16),
			activation="relu",
			max_iter=300,
			random_state=RANDOM_STATE,
		),
		{},
	),
	"svm": (SVC(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
	"xgb": (
		(
			XGBClassifier(
				use_label_encoder=False, eval_metric="logloss", n_jobs=-1, verbosity=0
			)
			if XGBClassifier
			else None
		),
		{"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.1, 0.3]},
	),
}
results = []


# ---------------------------------------------------------------------------
# Main procedure
# ---------------------------------------------------------------------------


def main(csv_path: Path, model_key: str):
	DGH_MAP = {dgh.column_name: dgh for dgh in get_csv_banking_dghs()}
	df = pd.read_csv(csv_path).drop(columns=["index"])
	train_df, test_df = train_test_split(
		df,
		test_size=TEST_SIZE,
		random_state=RANDOM_STATE,
		stratify=df[TARGET_COL],
	)

	prep = Preprocessor(DGH_MAP, STRATEGY)
	prep.fit(train_df.drop(columns=[TARGET_COL]))
	X_train, y_train = encode_features(prep, train_df, y_col=TARGET_COL)
	X_test, y_test = encode_features(prep, test_df, y_col=TARGET_COL)

	model_base, param_grid = MODEL_REGISTRY[model_key]
	if model_base is None:
		raise ImportError("XGBoost not installed")

	if param_grid:
		model = GridSearchCV(
			model_base, param_grid, cv=CV, scoring="accuracy", n_jobs=-1
		)
		model.fit(X_train, y_train)
		best_model = model.best_estimator_
		print(f"{model_key}: Best CV params: {model.best_params_}")
	else:
		best_model = model_base.fit(X_train, y_train)
		print(f"{model_key}: No hyperparameter search; using default.")

	preds = best_model.predict(X_test)
	acc = accuracy_score(y_test, preds)
	report = classification_report(y_test, preds, output_dict=True)
	print(f"Test Accuracy: {acc:.4f}\n")
	print(classification_report(y_test, preds))

	result_entry = {
		"model": model_key,
		"k": int(csv_path.stem.split("_k")[-1]),
		"accuracy": acc,
		"precision_0": report["no"]["precision"],
		"recall_0": report["no"]["recall"],
		"f1_0": report["no"]["f1-score"],
		"precision_1": report["yes"]["precision"],
		"recall_1": report["yes"]["recall"],
		"f1_1": report["yes"]["f1-score"],
		"macro_avg_precision": report["macro avg"]["precision"],
		"macro_avg_recall": report["macro avg"]["recall"],
		"macro_avg_f1": report["macro avg"]["f1-score"],
		"weighted_avg_precision": report["weighted avg"]["precision"],
		"weighted_avg_recall": report["weighted avg"]["recall"],
		"weighted_avg_f1": report["weighted avg"]["f1-score"],
	}
	results.append(result_entry)

	return acc


if __name__ == "__main__":
	k_values = [2 ** i for i in range(11)]
	model_keys = ["logistic", "tree", "forest", "mlp", "svm"]

	if XGBClassifier:
		# model_keys.append("xgb")
		pass

	for model_key in model_keys:
		print(f"=== Model: {model_key} ===")
		for i in k_values:
			print(f"k={i}")
			csv_path = (
				Path("../anjana_anonymizer")
				/ f"banking_simplified_anonymized_clean/banking_k{i}.csv"
			)
			main(csv_path, model_key)
		print("\n")

	with open("banking_model_results.csv", "w", newline="") as f:
		fieldnames = sorted({k for r in results for k in r})
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(results)
