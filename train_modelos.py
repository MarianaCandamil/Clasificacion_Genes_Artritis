#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# -------------------------------------------------------------
# 1. Cargar CSV
# -------------------------------------------------------------
csv_file = "subset_balanceado.csv"

print(f"➡ Cargando archivo: {csv_file}")
df = pd.read_csv(csv_file)

# -------------------------------------------------------------
# 2. Convertir la columna "encoded" de string → vector numérico
# -------------------------------------------------------------
def text_to_vector(text):
    # Convierte "1 0 0 1 0 ..." a lista [1,0,0,1,0...]
    return np.array(list(map(int, text.split())))

print("➡ Convirtiendo secuencias codificadas a vectores numéricos...")
X = np.vstack(df["encoded"].apply(text_to_vector).values)

y = df["label"].values

print(f"✔ Shape de X: {X.shape}")
print(f"✔ Shape de y: {y.shape}")

# -------------------------------------------------------------
# 3. División en entrenamiento y prueba
# -------------------------------------------------------------
print("➡ Dividiendo datos en train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# -------------------------------------------------------------
# 4. Entrenar Logistic Regression
# -------------------------------------------------------------
print("\n Entrenando Logistic Regression...")

logreg = LogisticRegression(
    max_iter=200,
    n_jobs=-1,
    solver="lbfgs"
)

logreg.fit(X_train, y_train)

# Predicciones
y_pred_lr = logreg.predict(X_test)
y_prob_lr = logreg.predict_proba(X_test)[:, 1]

# Métricas
acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)

print("✔ Logistic Regression Accuracy:", acc_lr)
print("✔ Logistic Regression AUC:", auc_lr)
print("✔ Logistic Regression Matriz de confusión:\n", cm_lr)

# Guardar modelo
joblib.dump(logreg, "modelo_logistic_regression.pkl")
print("💾 Modelo guardado: modelo_logistic_regression.pkl")

# -------------------------------------------------------------
# 5. Entrenar XGBoost
# -------------------------------------------------------------
print("\n Entrenando XGBoost...")

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    n_jobs=-1
)

xgb.fit(X_train, y_train)

# Predicciones
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

# Métricas
acc_xgb = accuracy_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_prob_xgb)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

print("✔ XGBoost Accuracy:", acc_xgb)
print("✔ XGBoost AUC:", auc_xgb)
print("✔ XGBoost Matriz de confusión:\n", cm_xgb)

# Guardar modelo
joblib.dump(xgb, "modelo_xgboost.pkl")
print(" Modelo guardado: modelo_xgboost.pkl")



# -------------------------------------------------------------
# 6. Entrenar Random Forest
# -------------------------------------------------------------
print("\n Entrenando Random Forest...")


rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

# Predicciones
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Métricas
acc_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

print("✔ Random Forest Accuracy:", acc_rf)
print("✔ Random Forest AUC:", auc_rf)
print("✔ Random Forest Matriz de confusión:\n", cm_rf)

# Guardar modelo
joblib.dump(rf, "modelo_random_forest.pkl")
print("💾 Modelo guardado: modelo_random_forest.pkl")




print("\n ¡Proceso completado!")

