# Respuestas al enunciado (ancladas a resultados del run)

## Punto 1 — BiLSTM / perplejidad

### 1. ¿Qué significa una perplejidad de X?

La perplejidad es \(\exp(-\mathbb{E}[\log p(w_{correcta}\mid contexto)])\).  
En el run final (`./results/punto1_bilstm.json`): **X ≈ 416**. El modelo se comporta *en promedio* como si eligiera entre ~416 alternativas equiprobables del vocabulario en cada paso. Es la exponencial de la entropía cruzada media: cuanto más probabilidad asigne a la palabra correcta, menor es la “sorpresa”. Valor **< 500** (aceptable); para acercarse a **< 100** harían falta más datos/épocas/capacidad.

**Si fuera alta (>500), hiperparámetros a modificar:**

1. **`EMBEDDING_DIM` (200→300):** mejor semántica ⇒ mayor \(P\) de la palabra correcta (ya aplicado en el reentrenamiento).  
2. **`MAX_LEN` / unidades LSTM:** más contexto y capacidad ⇒ menos incertidumbre condicional.  
3. Más datos limpios y/o más épocas con Early Stopping (evitar sobreajuste).

### 2. `predict_next_word` — tres tipos

Ver `./results/punto1_predicciones.json` (run final):

- **(a)** `Buenos días cómo` → **`se`** (continuación gramatical frecuente).  
- **(b)** `El banco está cerca del` → **`hotel`** (coocurrencia espacial; no desambigua “banco” financiero vs asiento).  
- **(c)** jerga/OOV → predicción genérica del dominio (`hotel`); las OOVs diluyen la señal del contexto.

### 3. Early Stopping y overfitting

Run final: **9 épocas**, mejor checkpoint en época **5** (`val_loss` mínima).  
Train loss seguía bajando mientras val_loss subía desde la época 6 ⇒ brecha clásica de overfitting.  
Sin Early Stopping hasta 80 épocas, la perplejidad de **test** tendería a **empeorar** tras el óptimo de validación.
---

## Punto 2 — Embeddings / visualización

1. **Word2Vec vs FastText:** W2V es más “discreto” y falla en OOV; FastText (n-gramas de caracteres) es más robusto a morfología española y produce nubes más densas/continuas.  
2. **t-SNE vs PCA (dim=300→2D):** PCA lineal (varianza global); t-SNE no lineal (vecindarios locales, mejor separación visual de clusters).  
3. **Linealidad:** PCA ordena por ejes de máxima varianza; t-SNE “pliega” el espacio para preservar similitudes locales (distancias globales no son métricas fieles).

Figuras: `./figures/tsne_*.png`, `pca_*.png`.

---

## Punto 5 — Análisis final

### ¿Por qué BETO ~92% y XLM-R ~93% pese a la complejidad del sarcasmo?

En este run: **BETO F1≈0.916 (bs=8)** y **XLM-R-large F1≈0.934 (bs=16)** en validación.  
El dataset es relativamente homogéneo (patrones irónicos léxicos recurrentes) y los transformers capturan contexto bidireccional / ironía superficial mejor que n-gramas.

### ¿Por qué LogReg (~85.8% / este run F1_macro≈0.835) es menor?

TF-IDF + LogReg solo ve n-gramas superficiales; no modela composición contextual ni tono. Los transformers sí, vía atención multicapa.

### ¿TF-IDF vs atención?

- **TF-IDF:** bolsa de n-gramas ponderada, sin orden profundo.  
- **Atención:** pesos dinámicos entre tokens según contexto; captura dependencias a distancia.

### ¿Efecto del batch size?

| Modelo | bs=8 | bs=16 | bs=32 |
| --- | --- | --- | --- |
| BETO | **F1 0.916** | 0.908 | 0.913 |
| XLM-R-large | falló/no aprendió (0.33) | **F1 0.934** | 0.931 |

Batch pequeño ⇒ más ruido, a veces mejor generalización, más pasos. Batch grande ⇒ más estable pero más VRAM (XLM-R-large). El mejor batch se eligió por F1 de validación.

### ¿Bastan los datos de Sarcasmo?

Sí para una **línea base fuerte** en este dominio etiquetado. Sarcasmo abierto (nuevos dominios, multimodal, conversacional) requeriría más datos y contexto.

### Test de 30 sentencias (20 sarcásticas / 10 no)

| Modelo | F1 macro | Accuracy |
| --- | ---: | ---: |
| LogReg TF-IDF | 0.824 | 0.833 |
| BETO | 0.861 | 0.867 |
| XLM-RoBERTa-large | **0.928** | **0.933** |

Detalle: `./results/punto5_test30.json`, `./results/test30_sentences.csv`.
