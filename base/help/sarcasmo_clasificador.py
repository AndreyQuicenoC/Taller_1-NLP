# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Clasificador de Sarcasmo en Español — Regresión Logística                  ║
# ║  Dataset: Ernesto-1997/Sarcastic_spanish_dataset (HuggingFace)              ║
# ║  Pipeline: TF-IDF + Logistic Regression | métricas F1 | ventana de test     ║
# ║  EISC — Universidad del Valle                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Instalación:
#   pip install datasets scikit-learn streamlit

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ID   = "Ernesto-1997/Sarcastic_spanish_dataset"
MODEL_CACHE  = "sarcasmo_model.pkl"   # guarda el pipeline entrenado

st.set_page_config(
    page_title="Clasificador de Sarcasmo 🙃",
    page_icon="🙃",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Cargando dataset y entrenando modelo…")
def train_pipeline():
    from datasets import load_dataset
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        classification_report, f1_score,
        confusion_matrix, ConfusionMatrixDisplay,
    )
    import matplotlib.pyplot as plt

    # ── 1. Cargar datos ───────────────────────────────────────────────────────
    ds = load_dataset(DATASET_ID)
    train_df = ds["train"].to_pandas()
    val_df   = ds["validation"].to_pandas()

    X_train, y_train = train_df["Texto"].astype(str), train_df["Sarcasmo"].astype(int)
    X_val,   y_val   = val_df["Texto"].astype(str),   val_df["Sarcasmo"].astype(int)

    # ── 2. Pipeline TF-IDF + Regresión Logística ──────────────────────────────
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer      = "word",
            ngram_range   = (1, 2),     # unigramas + bigramas
            max_features  = 50_000,
            sublinear_tf  = True,       # log(TF) → reduce sesgo de términos frecuentes
            strip_accents = "unicode",
            min_df        = 2,
        )),
        ("clf", LogisticRegression(
            C             = 5.0,        # regularización suave
            max_iter      = 1000,
            solver        = "lbfgs",
            class_weight  = "balanced", # maneja desbalance de clases
            n_jobs        = -1,
        )),
    ])

    # ── 3. Entrenamiento ──────────────────────────────────────────────────────
    pipe.fit(X_train, y_train)

    # ── 4. Evaluación ─────────────────────────────────────────────────────────
    y_pred  = pipe.predict(X_val)
    y_prob  = pipe.predict_proba(X_val)[:, 1]

    report  = classification_report(
        y_val, y_pred,
        target_names=["No sarcástico", "Sarcástico"],
        output_dict=True,
    )
    f1_macro = f1_score(y_val, y_pred, average="macro")
    f1_w     = f1_score(y_val, y_pred, average="weighted")

    # ── 5. Matriz de confusión ────────────────────────────────────────────────
    cm = confusion_matrix(y_val, y_pred)
    fig_cm, ax = plt.subplots(figsize=(4, 3.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No sarcástico", "Sarcástico"])
    disp.plot(ax=ax, colorbar=False, cmap="Reds")
    ax.set_title("Matriz de confusión — validación")
    plt.tight_layout()

    # ── 6. Top palabras por clase ─────────────────────────────────────────────
    feature_names = np.array(pipe["tfidf"].get_feature_names_out())
    coefs         = pipe["clf"].coef_[0]
    top_n         = 15
    top_sarcastic = feature_names[np.argsort(coefs)[-top_n:][::-1]]
    top_normal    = feature_names[np.argsort(coefs)[:top_n]]

    # ── 7. Guardar pipeline ───────────────────────────────────────────────────
    with open(MODEL_CACHE, "wb") as f:
        pickle.dump(pipe, f)

    return {
        "pipe":           pipe,
        "report":         report,
        "f1_macro":       f1_macro,
        "f1_weighted":    f1_w,
        "fig_cm":         fig_cm,
        "top_sarcastic":  top_sarcastic,
        "top_normal":     top_normal,
        "n_train":        len(X_train),
        "n_val":          len(X_val),
    }


results = train_pipeline()
pipe    = results["pipe"]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — resumen de métricas globales (visible desde cualquier pestaña)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Métricas globales")
    st.caption("Evaluadas sobre el split de **validación** (3 820 ejemplos)")
    st.divider()

    r = results["report"]

    st.metric("F1 Macro",      f"{results['f1_macro']*100:.2f}%",
              help="Media no ponderada del F1 de cada clase")
    st.metric("F1 Ponderado",  f"{results['f1_weighted']*100:.2f}%",
              help="F1 ponderado por el número de ejemplos por clase")
    st.divider()

    st.markdown("**Por clase**")
    col_a, col_b = st.columns(2)
    col_a.metric("Sarcástico 🙃",
                 f"{r['Sarcástico']['f1-score']*100:.2f}%",
                 help="F1 de la clase positiva")
    col_b.metric("No sarcástico 😐",
                 f"{r['No sarcástico']['f1-score']*100:.2f}%",
                 help="F1 de la clase negativa")
    st.divider()

    st.markdown("**Precisión y Recall**")
    detail = {
        "Clase": ["Sarcástico 🙃", "No sarcástico 😐"],
        "Precisión": [
            f"{r['Sarcástico']['precision']*100:.1f}%",
            f"{r['No sarcástico']['precision']*100:.1f}%",
        ],
        "Recall": [
            f"{r['Sarcástico']['recall']*100:.1f}%",
            f"{r['No sarcástico']['recall']*100:.1f}%",
        ],
    }
    st.dataframe(pd.DataFrame(detail), hide_index=True, use_container_width=True)
    st.divider()

    st.caption(f"🏋️ Train: {results['n_train']:,} · ✅ Val: {results['n_val']:,}")
    st.caption("Modelo: TF-IDF (1–2g) + LogReg · C=5 · balanced")

# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ
# ─────────────────────────────────────────────────────────────────────────────
st.title("🙃 Clasificador de Sarcasmo en Español")
st.caption("Regresión Logística · TF-IDF (1–2 gramas) · Dataset: Ernesto-1997/Sarcastic_spanish_dataset")

tab1, tab2, tab3 = st.tabs(["🔍 Probar texto", "📊 Métricas", "🔤 Palabras clave"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — VENTANA DE PRUEBA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Escribe o pega un texto en español")

    ejemplos = [
        "Sí claro, como siempre llegaste a tiempo, qué maravilla.",
        "El partido estuvo increíble, jugaron de lujo.",
        "¡Wow, qué sorpresa! Nunca pensé que llegarías tarde OTRA VEZ.",
        "Me encanta madrugar los lunes, es lo mejor del mundo.",
        "La reunión fue corta y productiva.",
    ]
    ejemplo_sel = st.selectbox(
        "Ejemplos rápidos (opcional):",
        ["— selecciona un ejemplo —"] + ejemplos,
    )

    texto_default = "" if ejemplo_sel == "— selecciona un ejemplo —" else ejemplo_sel
    texto = st.text_area(
        "Texto a clasificar:",
        value   = texto_default,
        height  = 130,
        placeholder="Escribe aquí cualquier frase en español…",
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        clasificar = st.button("Clasificar 🚀", type="primary", use_container_width=True)

    if clasificar:
        if not texto.strip():
            st.warning("Escribe algún texto primero.")
        else:
            prob  = pipe.predict_proba([texto])[0]
            pred  = int(np.argmax(prob))
            conf  = prob[pred] * 100

            if pred == 1:
                st.error(f"### 🙃 Sarcástico  ({conf:.1f}% confianza)")
            else:
                st.success(f"### 😐 No sarcástico  ({conf:.1f}% confianza)")

            # Barra de probabilidades — una columna por clase para poder colorear
            st.markdown("**Distribución de probabilidades:**")
            prob_df = pd.DataFrame({
                "No sarcástico 😐": [prob[0]],
                "Sarcástico 🙃":    [prob[1]],
            })
            st.bar_chart(prob_df, color=["#4CAF50", "#F44336"])

    # Clasificación en lote
    st.divider()
    st.subheader("Clasificación en lote")
    batch_texto = st.text_area(
        "Un texto por línea:",
        height      = 160,
        placeholder = "Frase 1\nFrase 2\nFrase 3…",
        key         = "batch",
    )
    if st.button("Clasificar todas 📋"):
        lineas = [l.strip() for l in batch_texto.splitlines() if l.strip()]
        if not lineas:
            st.warning("Escribe al menos una línea.")
        else:
            probs  = pipe.predict_proba(lineas)
            preds  = np.argmax(probs, axis=1)
            labels = ["🙃 Sarcástico" if p == 1 else "😐 No sarcástico" for p in preds]
            confs  = [f"{probs[i, preds[i]]*100:.1f}%" for i in range(len(lineas))]
            df_res = pd.DataFrame({
                "Texto":       lineas,
                "Predicción":  labels,
                "Confianza":   confs,
            })
            st.dataframe(df_res, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Resultados en el conjunto de validación")

    r = results["report"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("F1 Macro",    f"{results['f1_macro']*100:.2f}%")
    col2.metric("F1 Ponderado", f"{results['f1_weighted']*100:.2f}%")
    col3.metric("F1 Sarcástico",     f"{r['Sarcástico']['f1-score']*100:.2f}%")
    col4.metric("F1 No sarcástico",  f"{r['No sarcástico']['f1-score']*100:.2f}%")

    st.divider()
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**Reporte completo**")
        report_df = pd.DataFrame(results["report"]).T.drop("accuracy", errors="ignore")
        report_df = report_df[["precision", "recall", "f1-score", "support"]].round(4)
        st.dataframe(report_df, use_container_width=True)

        info_cols = st.columns(2)
        info_cols[0].info(f"🏋️ Train: {results['n_train']:,} ejemplos")
        info_cols[1].info(f"✅ Val: {results['n_val']:,} ejemplos")

    with col_b:
        st.markdown("**Matriz de confusión**")
        st.pyplot(results["fig_cm"])

    st.divider()
    st.subheader("Parámetros del modelo")
    params = {
        "Vectorizador":    "TF-IDF",
        "N-gramas":        "(1, 2)",
        "Max features":    "50 000",
        "Sublinear TF":    "Sí",
        "Regularización C": "5.0",
        "Solver":          "lbfgs",
        "Class weight":    "balanced",
        "Max iter":        "1 000",
    }
    st.table(pd.DataFrame(params.items(), columns=["Parámetro", "Valor"]))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PALABRAS CLAVE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Palabras más indicativas por clase")
    st.caption(
        "Los coeficientes del modelo muestran qué n-gramas empujan hacia cada clase. "
        "Valores positivos altos → sarcasmo; negativos → no sarcasmo."
    )

    col_s, col_n = st.columns(2)

    with col_s:
        st.markdown("#### 🙃 Sarcástico")
        coefs_sarc = np.array(sorted(pipe["clf"].coef_[0])[-15:][::-1]).round(4)
        df_sarc = pd.DataFrame({
            "N-grama":      results["top_sarcastic"],
            "Coeficiente":  coefs_sarc,
        })
        st.dataframe(df_sarc, use_container_width=True, hide_index=True)

    with col_n:
        st.markdown("#### 😐 No sarcástico")
        coefs_norm = np.array(sorted(pipe["clf"].coef_[0])[:15]).round(4)
        df_norm = pd.DataFrame({
            "N-grama":      results["top_normal"],
            "Coeficiente":  coefs_norm,
        })
        st.dataframe(df_norm, use_container_width=True, hide_index=True)
