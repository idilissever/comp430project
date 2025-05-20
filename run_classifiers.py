import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# XGBoost requires installation: pip install xgboost
from xgboost import XGBClassifier


def run_all_classifiers(df, target_column="income", test_size=0.2, random_state=42, show_report=True):
    """
    Runs Logistic Regression, Decision Tree, Random Forest, and XGBoost
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }

    for name, model in classifiers.items():
        print("=" * 50)
        print("Training:", name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print("✅ Accuracy: %.4f" % acc)
        if show_report:
            print(classification_report(y_test, preds, digits=4))


# Example usage
if __name__ == "__main__":
    from generalization_utils import preprocess_anonymized_dataframe

    df = pd.read_csv("anonymized_csv/mondrian_strict_k16.csv")

    df = preprocess_anonymized_dataframe(
        df,
        numeric_interval_columns=["age", "education_num"],
        categorical_columns=["workclass", "marital_status", "occupation", "race", "sex", "native_country"],
        target_column="income",
        keep_width=True
    )

    run_all_classifiers(df)
