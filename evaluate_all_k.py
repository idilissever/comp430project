import os
import pandas as pd
from generalization_utils import preprocess_anonymized_dataframe
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Config
INPUT_FOLDER = "basic_mondrian_adult_anonymized"
TARGET = "income"
NUMERIC_INTERVALS = ["age", "education_num"]
CATEGORICAL_COLS = ["workclass", "marital_status", "occupation", "race", "sex", "native_country"]

# Classifiers
MODELS = {
	"Logistic Regression": LogisticRegression(max_iter=1000),
	"Decision Tree": DecisionTreeClassifier(),
	"Random Forest": RandomForestClassifier(n_estimators=100),
	"XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = []

# Loop through all anonymized_k*.csv files
for filename in sorted(os.listdir(INPUT_FOLDER)):
	if not filename.endswith(".csv"):
		continue
	k_val = filename.split("_k")[-1].replace(".csv", "")
	df = pd.read_csv(os.path.join(INPUT_FOLDER, filename))

	print("=" * 60)
	print("Processing k =", k_val)

	df = preprocess_anonymized_dataframe(
		df,
		numeric_interval_columns=NUMERIC_INTERVALS,
		categorical_columns=CATEGORICAL_COLS,
		target_column=TARGET,
		keep_width=True
	)

	X = df.drop(columns=[TARGET])
	y = df[TARGET]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	for model_name, model in MODELS.items():
		model.fit(X_train, y_train)
		y_pred_train = model.predict(X_train)
		y_pred_test = model.predict(X_test)

		try:
			y_proba = model.predict_proba(X_test)[:, 1]
			auc = roc_auc_score(y_test, y_proba)
		except AttributeError:
			auc = None  # Model does not support predict_proba

		results.append({
			"k": int(k_val),
			"model": model_name,
			"train_accuracy": round(accuracy_score(y_train, y_pred_train), 4),
			"test_accuracy": round(accuracy_score(y_test, y_pred_test), 4),
			"precision": round(precision_score(y_test, y_pred_test), 4),
			"recall": round(recall_score(y_test, y_pred_test), 4),
			"f1_score": round(f1_score(y_test, y_pred_test), 4),
			"roc_auc_score": round(auc, 4) if auc is not None else None

		})
		print(f"✅ {model_name} @ k={k_val} → Test Accuracy: {round(accuracy_score(y_test, y_pred_test), 4):.4f}")

# Export summary report
report_df = pd.DataFrame(results)
report_df.sort_values(["model", "k"], inplace=True)
report_df.to_csv("classification_accuracy_report.csv", index=False)

print("\n🎉 Done! Accuracy report saved to classification_accuracy_report.csv")
