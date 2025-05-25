import streamlit as st
import pandas as pd
import joblib

# 1) Carga tu pipeline serializado
MODEL_PATH = "models/expenses_pipeline.pkl"

@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

pipeline = load_pipeline()

st.title("📊 Predicción de gastos estudiantiles")

# 2) Inputs numéricos (que no necesitan mapeo)
comidas_fuera = st.number_input("Comidas fuera de casa", min_value=0, step=1)
snacks_q      = st.number_input("Cantidad de snacks en Q: ",      min_value=0, step=1)
edad          = st.number_input("Edad : ",                    min_value=12, step=1)
materias_dia  = st.number_input("Materias al día",         min_value=0, step=1)
gasolina_q    = st.number_input("Gasto en gasolina (Q)",   min_value=0.0, step=0.1)

# 3) Extrae las categorías del OneHotEncoder del pipeline
#    pipeline.named_steps["prep"] es tu ColumnTransformer
#    transformers_[1][1] es el encoder (asumiendo que lo pusiste segundo)
ohe = pipeline.named_steps["prep"].transformers_[1][1]
cats = ohe.categories_  
# cats es una lista de arrays, en el mismo orden que cat_cols al entrenar:
#   cat_cols = ["lugar","transporte","actividades_extra","lleva_almuerzo",
#               "ocupacion","desayuno_casa","comparte_transporte","conduce"]

# 4) Usa esas categorías en tus selectboxes
lugar_opts      = list(cats[0])
transporte_opts = list(cats[1])
activ_extra_opts= list(cats[2])
almuerzo_opts   = list(cats[3])
ocupacion_opts  = list(cats[4])
desayuno_opts   = list(cats[5])
compartir_opts  = list(cats[6])
conduce_opts    = list(cats[7])

lugar             = st.selectbox("Lugar",               lugar_opts)
transporte        = st.selectbox("Medio de transporte", transporte_opts)
actividades_extra = st.selectbox("Actividades extra",   activ_extra_opts)
lleva_almuerzo    = st.selectbox("Lleva almuerzo",      almuerzo_opts)
ocupacion         = st.selectbox("Ocupación",           ocupacion_opts)
desayuno_casa     = st.selectbox("Desayuna en casa",    desayuno_opts)
comparte_transp   = st.selectbox("Comparte transporte", compartir_opts)
conduce           = st.selectbox("Conduce vehículo",    conduce_opts)

# 5) Botón de predicción
if st.button("Calcular gasto"):
    # 6) Construye el DataFrame en el mismo orden de columnas que entrenaste
    df_in = pd.DataFrame([{
        "lugar":              lugar,
        "transporte":         transporte,
        "actividades_extra":  actividades_extra,
        "lleva_almuerzo":     lleva_almuerzo,
        "ocupacion":          ocupacion,
        "desayuno_casa":      desayuno_casa,
        "comparte_transporte":comparte_transp,
        "conduce":            conduce,
        "comidas_fuera":      comidas_fuera,
        "snacks_q":           snacks_q,
        "edad":               edad,
        "materias_dia":       materias_dia,
        "gasolina_q":         gasolina_q
    }])
    # 7) Predice con el pipeline completo
    gasto_pred = pipeline.predict(df_in)[0]
    st.success(f"💰 Gasto estimado por día de clases: Q{gasto_pred:.2f}")
