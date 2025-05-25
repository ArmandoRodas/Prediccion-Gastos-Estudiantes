# train_pipeline.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import RidgeCV
import joblib

# 1) Carga tus datos (asegúrate de instalar openpyxl si es .xlsx)
df = pd.read_excel("data/datos_gasto_ampliado.xlsx", engine="openpyxl")

# 2) Limpia/renombra columnas
df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={"gasto_total_q": "gasto_total"})

# 3) Define X e y
cat_cols = ["lugar","transporte","actividades_extra","lleva_almuerzo",
            "ocupacion","desayuno_casa","comparte_transporte","conduce"]
num_cols = ["comidas_fuera","snacks_q","edad","materias_dia","gasolina_q"]
X = df[cat_cols + num_cols]
y = df["gasto_total"]

# 4) Monta el pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", sparse=False), cat_cols),
])
pipeline = Pipeline([
    ("prep", preprocessor),
    ("reg", RidgeCV(alphas=[0.1, 1.0, 10.0]))
])

# 5) Entrena
pipeline.fit(X, y)

# 6) Serializa el pipeline entrenado
joblib.dump(pipeline, "models/expenses_pipeline.pkl")
print("✔ Modelo entrenado y guardado en models/expenses_pipeline.pkl")
