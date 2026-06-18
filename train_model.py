import pandas as pd
import pickle
import joblib

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE



# LOAD DATASET


data = pd.read_csv("synthetic_stroke_data.csv")

print("\n===== DATASET PREVIEW =====")

print(data.head())

print("\n===== DATASET SHAPE =====")

print(data.shape)

print("\n===== COLUMN NAMES =====")

print(data.columns.tolist())



# SELECT REQUIRED COLUMNS


data = data[
    [
        'age',
        'hypertension',
        'heart_disease',
        'avg_glucose_level',
        'bmi',
        'stroke'
    ]
]



# REMOVE DUPLICATES


data = data.drop_duplicates()



# FEATURES AND TARGET


X = data[
    [
        'age',
        'hypertension',
        'heart_disease',
        'avg_glucose_level',
        'bmi'
    ]
]

y = data['stroke']



# HANDLE MISSING VALUES


imputer = SimpleImputer(
    strategy='mean'
)

X = imputer.fit_transform(X)



# FEATURE SCALING


scaler = StandardScaler()

X = scaler.fit_transform(X)



# HANDLE IMBALANCED DATASET


smote = SMOTE(
    random_state=42
)

X, y = smote.fit_resample(
    X,
    y
)

print("\n===== BALANCED DATASET SHAPE =====")

print(pd.Series(y).value_counts())



# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n===== TRAINING AND TESTING DATA =====")

print("Training Data :", len(X_train))

print("Testing Data  :", len(X_test))



# RANDOM FOREST MODEL


model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    n_jobs=-1

)

model.fit(
    X_train,
    y_train
)


# PREDICTIONS


y_pred = model.predict(
    X_test
)



# MODEL EVALUATION


accuracy = accuracy_score(
    y_test,
    y_pred
)

roc = roc_auc_score(
    y_test,
    y_pred
)

print("\n===== MODEL ACCURACY =====")

print(round(accuracy * 100, 2), "%")

print("\n===== ROC AUC SCORE =====")

print(round(roc * 100, 2), "%")

print("\n===== CONFUSION MATRIX =====")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# SAVE FILES


model = joblib.load("stroke_model_compressed.joblib")

pickle.dump(
    imputer,
    open("imputer.pkl", "wb")
)

pickle.dump(
    scaler,
    open("scaler.pkl", "wb")
)

print("\n===== MODEL SAVED SUCCESSFULLY =====")