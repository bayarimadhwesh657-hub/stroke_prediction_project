import pandas as pd


# Load Dataset

df = pd.read_csv("synthetic_stroke_data.csv")

print("Dataset Shape:", df.shape)

# Select Features

X = df[
    [
        'age',
        'hypertension',
        'heart_disease',
        'avg_glucose_level',
        'bmi'
    ]
]

y = df['stroke']

# Data Cleaning


X = X.fillna(X.mean())

# Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

# Apply SMOTE

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X, y = smote.fit_resample(X, y)

print("\nBalanced Dataset")

print(pd.Series(y).value_counts())

# Train Test Split

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Models

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

models = {

    "Logistic Regression":

        LogisticRegression(max_iter=1000),

    "Decision Tree":

        DecisionTreeClassifier(random_state=42),

    "Random Forest":

        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}

# Metrics

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Evaluate Models

results = []

for name, model in models.items():

    model.fit(xtrain, ytrain)

    ypred = model.predict(xtest)

    accuracy = accuracy_score(ytest, ypred)

    precision = precision_score(ytest, ypred)

    recall = recall_score(ytest, ypred)

    f1 = f1_score(ytest, ypred)

    roc = roc_auc_score(ytest, ypred)

    print("\n=====================")

    print("MODEL:", name)

    print("=====================")

    print("\nConfusion Matrix")

    print(confusion_matrix(ytest, ypred))

    print("\nAccuracy :", round(accuracy*100, 2), "%")

    print("Precision:", round(precision*100, 2), "%")

    print("Recall   :", round(recall*100, 2), "%")

    print("F1 Score :", round(f1*100, 2), "%")

    print("ROC AUC  :", round(roc*100, 2), "%")

    results.append(
        [name, accuracy]
    )

# Best Model

best_model = max(
    results,
    key=lambda x: x[1]
)

print("\n=====================")

print("BEST MODEL")

print("=====================")

print(best_model[0])

print("Accuracy:", round(best_model[1]*100, 2), "%")