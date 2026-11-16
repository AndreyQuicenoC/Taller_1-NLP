Taller 1. Introducción de conceptos; LSTMs, tokenizadores,

datasets, embedding y análisis de sentimientos

1.(MACHINE LEARNING-LSTMs) El objetivo de éste punto es experimentar con modelos de Redes Neuronales Recurrentes (RNNs), específicamente LSTM (Long Short-Term Memory) y BiLSTM (Bidirectional LSTM), para realizar el modelado del lenguaje en español. El modelo debe predecir la palabra siguiente dada una secuencia de palabras (oración o sentencia), utilizando técnicas modernas de procesamiento del lenguaje natural (NLP).

Este ejercicio permite explorar cómo los modelos de etiquetado secuencial aprenden patrones lingüísticos, como dependencias temporales, contexto y semántica, a partir de datos textuales. Para entrenar los modelos tenemos los siguientes componentes:

Componentes del proyecto

## 1.1 Carga del Dataset

- Se utilizará el dataset spanish_billion_words_clean disponible en Hugging Face. [URL 🔗](https://huggingface.co/datasets/jhonparra18/spanish_billion_words_clean)

- Este corpus contiene millones de oraciones en español, lo que permite entrenar modelos robustos.

- Si se puede cargar completo el dataset, mejor.

from datasets import load_dataset

```
dataset = load_dataset("text", data_dir="path/to/spanish_billion_words_clean") sentences = dataset["train"]
```

## 1.2 Tokenización y Creación del Vocabulario

- 1) Se aplicará la tokenización de palabras usando Tokenizer de Keras.

- 2) Se generará un vocabulario de palabras únicas.

- 3) Cada palabra se mapea a un índice entero (word_index).

- 4) Se crea un diccionario inverso (index_word) para reconstruir texto.

tokenizer = Tokenizer(oov_token="<OOV>") (Tokenización por palabras)

tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)


vocab_size = len(tokenizer.word_index) + 1

## 1.3 Creación del Conjunto de Entrenamiento (X, Y)

- X: Secuencias de entrada (lista de listas de índices)

- Y: Índice de la palabra siguiente (no one-hot, ya que usamos sparse_categorical_crossentropy)

- Para cada secuencia [w1, w2, ..., wn], se crean pares:

- (w1) → w2

- (w1, w2) → w3

- ...

- (w1, ..., wn-1) → wn

## 1.4 Padding y Truncado

- Se calcula MAX_LEN = max(len(x) for x in X)

- Se aplica padding (pad_sequences) para alinear todas las secuencias a MAX_LEN.

- Se puede truncar si MAX_LEN es muy alto (> 50) para mejorar eficiencia.

MAX_LEN = min(50, max(len(x) for x in X))

X_padded = pad_sequences(X, maxlen=MAX_LEN, padding='pre')

## 1.5 División del Conjunto (Train/Test)

- División en 80% entrenamiento / 20% prueba.

- Uso de train_test_split:

## 1.6 Construcción del Modelo de LSTM o BiLSTM

Construcción del modelo secuencia del LSTM, para ello podemos usar una dimensión de embedding de 100 (o más hasta 300) y 128 unidades para LSTMs. La función de activación ReLU (Rectified Linear Unit) es rápida y no satura como las funciones sigmoide y tanh. La Dense(64) con la ReLU reducen gradualmente la dimensionalidad y permiten actuar en un espacio más “pequeño”. Esta función de activación toma la salida de la capa anterior de 64 o 128 y le aplica una transformación lineal y produce un vector de 64 valores que van a la capa final Dense(vocab_size, softmax). La creación del modelo también tiene el modo de

compilación, en este caso usamos

sparse_categorical_crossentropy que no necesita one-hot para la Y, también usaremos el optimizador de adam.

la función de pérdida de


## 1.7 Entrenamiento del modelo con Early Stopping

El entrenamiento del dataset (X,Y), en la que se tienen en cuenta las épocas de entrenamiento y el batch_size que es el tamaño de las muestras. Para evitar el overfitting o sobreajuste se debe definir la técnica de Early stopping para la prevención del overfitting el cuál detiene el entrenamiento del modelo antes que comience a sobreajustarse.

```
early_stopping = Earlystopping(
monitor="val_loss", $ < Métrica que vigila
patience=15, $ < Número de épocas sin mejora antes de parar
restore best _weights=True, $ Recupera Los pesos del mejor modelo
verbose=1 < Muestra mensaje cuando se detiene
)
EPOCHS = 3
BATCH_SIZE = 128
modelo BiLSTM...")
history = model.fit(
X train, y train, <-- etiquetas como enteros
epochs=EPOCHS,
batch_size=BATCH_SIZE,
lation_split=0.1,
stopping 4 < Aquí va el callback
```

## 1.8 Cálculo de la perplejidad

El cálculo de la perplejidad del conjunto de testeo es una de las métricas más importantes en modelamiento del lenguaje, para la predicción de la palabra siguiente. Si el modelo siempre está muy seguro y acierta fácilmente → perplejidad baja → ¡buen modelo. Si el modelo siempre está confundido y le parece que cualquier palabra podría venir → perplejidad alta → ¡mal modelo!. En términos generales, la perplejidad mide la sorpresa del modelo ante la palabra correcta. Para cada palabra real, se toma la probabilidad que el modelo le asignó (de entre todo el vocabulario), se calcula el logaritmo para suavizarla, se promedia sobre todo el conjunto de prueba, y finalmente la perplejidad es el exponencial del negativo de ese promedio.

- . [ws 1) = probabilidad que el modelo asigna a la palabra correcta 1, en la posición £

Entonces:

Donde N es el número total de palabras del conjunto de testeo. La siguiente es la implementación de la perplejidad del conjunto de testeo.


```
import numpy as mp
def calculate X test, y test, tokenizer):
probs = model.predict(X test, verbose=0)
log_probs = np.log(probs[np.arange(len(y_test)), y_test])
avg_log_prob = np.mean(log_probs)
perplexity = np.exp(-avg_log_prob)
return perplexity
```

Estos son algunos valores de perplejidad con su descripción.

## 1.9 Predicción de la próxima palabra

Predicción de la próxima palabra dada una sentencia X dada por el usuario, en este caso se crea un procedimiento para predecir la próxima palabra predict_next_word(model, tokenizer, user_input, MAX_LEN) que tenga en cuenta el modelo , la tokenización y la enterización de la sentencia de entrada.

Para la entrega de los resultados se deben tener en cuenta la información de las dos siguientes tablas:

*Tabla 1 Resultados esperados*

| Modelo | Perplejidad | Observaciones |
| --- | --- | --- |
| LSTMs | < 100 es muy buen resultado | - Baja perplejidad → modelo seguro |
|   | y >500 ; hay que usar más | y acierto alto → buen trabajo! |
|   | épocas, usar más datos, | - Alta perplejidad → modelo |
|   | aumentar el embedding del | confundido → necesita más datos, |


| LSTM (100-300) y también la | más épocas o ajustes. |
| --- | --- |
| longitud de las secuencias, es |   |
| decir aumentar el MAX_LEN |   |
| BiLSTMs Similar o mejor que LSTM | - Capta contexto bidireccional → |
|   | mejor rendimiento en tareas |
|   | complejas. |
|   | - Requiere más recursos |
|   | computacionales. |

*Tabla 2. Parámetros de todo el proceso de predicción*

| Parámetros | Recomendación |
| --- | --- |
| Tamaño del dataset | Usar más datos (≥ 50,000 oraciones) |
|   | para mejorar generalización |
| Dimensión del embedding | Probar entre 100 y 300 (mejor para |
|   | capturar semántica) |
| MAX_LEN | Ajustar según longitud promedio de |
|   | oraciones (ej: 40–60) |
| Unidades LSTM | Probar 64, 128, 256; 64 es suficiente |
|   | para datos pequeños |
| Dropout (opcional) | Añadir |
|   | dropout=0.2 |
|   | para evitar overfitting |
| Batch size | 32, 64, 128 |
| Épocas | 30, 50 , 80 (dependiendo del tamaño |
|   | del dataset) Cuando se tienen pocos |
|   | datos en el dataset es mejor repetirlos |
|   | con más épocas. Se entendería que |
|   | con más datos el modelo aprende más |
|   | rápido y por ello es mejor usar menos |
|   | épocas. |

Como producto final se debe entrenar un BiLSTM con todas las características antes mencionadas y obtener el menor valor de perplejidad. La idea también es la de responder las siguientes preguntas:


- 1. ¿Qué significa realmente, en términos de la distribución de probabilidad sobre el vocabulario, que tu modelo tenga una perplejidad de X? Si tu perplejidad fue alta, identifica al menos dos hiperparámetros específicos de tu arquitectura (ej. dimensión del embedding, MAX_LEN, unidades LSTM) que modificarías y justifica cómo cada cambio reduciría la "sorpresa" del modelo ante la palabra correcta.

- 2. Prueba tu función predict_next_word() con al menos tres tipos de oraciones: (a) una frase muy común del español (ej. "Buenos días, ¿cómo"), (b) una oración con ambigüedad semántica (ej. "El banco está cerca del"), (c) una oración con jerga, neologismos o palabras fuera del vocabulario (OOV).

- 3. Observando las curvas de pérdida (loss) de entrenamiento y validación de tu modelo, ¿en qué época se detuvo el entrenamiento y por qué? ¿Identificaste una brecha creciente entre la pérdida de entrenamiento y la de validación (señal clásica de overfitting)? Si hubieras entrenado sin Early Stopping durante las 80 épocas completas, ¿cómo cree que habría cambiado la perplejidad en el conjunto de prueba?

- 2. EMBEDDINS DE PALABRAS Sobre el dataset spanish_billion_words entrenar los modelos de Word2vec y FastText sobre Gensim con el fin de obtener un modelo de embedding de palabras. Sobre los dos modelos entrenados se debe implementar los siguientes pasos: [URL 🔗](https://huggingface.co/datasets/jhonparra18/spanish_billion_words_clean)

## 2.1 Preprocesamiento del corpus

- Cargar el dataset spanish_billion_words luego preprocesarlo [URL 🔗](https://huggingface.co/datasets/jhonparra18/spanish_billion_words_clean)

- Tokenizar cada una de las sentencias del dataset y luego aplicar las siguientes transformaciones:

- Eliminación de números.

- Eliminación de signos de puntuación y símbolos especiales.

- Eliminación de palabras vacías (stopwords) del español.

- Tokenización de las sentencias en listas de palabras.

- La idea es la de obtener una lista de listas de las palabras tokenizadas y limpias del dataset.

## 2.2 Entrenamiento de modelos

Entrenar los siguientes modelos con las configuraciones indicadas:

| Modelo | Tamaño de embeddings | Tamaño de sentencias |
| --- | --- | --- |
| Word2vec | 100, 200 y 300 | 500000, 5 y 10 Millones |


| FastText | 100, 200 y 300 | 500000, 5 y 10 Millones |
| --- | --- | --- |

## 2.3 Consulta de similitud semántica

Implementar una función que dada una palabra de entrada, devuelva las 10 palabras más similares según el modelo seleccionado teniendo en cuenta el mejor modelo de los dos por sus respectivas características.

## 2.4 Creación del archivo de vectores

Entrenar dos modelos de embeddings —Word2Vec y FastText— sobre el dataset generando cada uno un archivo con vectores de palabras de dimensión 300 en formato texto plano.

## 2.5 Visualización con t-SNE y PCA

Visualizar en el plano cartesiano (Tsne y PCA) los vectores de embeddings entrenados con los modelos de Wordvec y Fasttext para los tamaños de sentencias de 50, 100 y 200 también se pueden escoger por número de palabras; 50-100. Comparar las visualizaciones de las palabras en el plano cartesiano, según los dos modelos del embedding de 300. Para éste ejercicio se debe explicar: [URL 🔗](https://radimrehurek.com/gensim/models/word2vec.html)

- 1)¿Cuál es la diferencia en las distribuciones de los dos embedder para cada uno de los tamaños de secuencias?

- 2) ¿Cuál es la diferencia entre Tsne y PCA en la visualización de los conjuntos de sentencias con Dim=300 ? [URL 🔗](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

- 3) ¿Cuál es la incidencia de la linealidad y no linealidad de los métodos PCA y Tsne en su orden con respecto a la visualización de las palabras?

## 3. Entrenamiento de un Tokenizador con SentencePiece

Dataset : spanish_billion_words (texto no anotado) [URL 🔗](https://huggingface.co/datasets/jhonparra18/spanish_billion_words_clean)

## Tareas :

- 1. Entrenar un tokenizador con SentencePiece para vocabularios de tamaño 5k, 10k y 20k , usando:

- Modelo tipo "word" (sin segmentación de subpalabras).

- Símbolos definidos por el usuario : <sep>, <cls/>.

- Símbolos definidos por el usuario : <s>, </s>.

- 2. Parámetros de entrenamiento :


- model_prefix: Prefijo para los archivos generados.

- input: Ruta del dataset (ej: wiki/*.txt).

- 3. Tokenizar una sentencia de entrada (ejemplo: "El cáncer de mama es una enfermedad compleja" ) para cada modelo entrenado. Comparar con el tokenizador de T5 [URL 🔗](https://huggingface.co/google-t5/t5-base)

## 4 Tokenización con BETO,, XML-Roberta-large y GPT2 para el Dataset BioBERT [URL 🔗](https://huggingface.co/FacebookAI/xlm-roberta-large)

Dataset : BioBERT (anotado, con características ["sentencia", "tag"]). Tener en cuenta que este datasets se encuentra de manera local. [URL 🔗](https://drive.google.com/drive/u/0/folders/1GlTZm2UoAR2ANsLvaSdeuUT-ZLu3l5w9)

## Tareas :

- 4.1. Cargar los conjuntos train, validation y test usando load_dataset("Biobert_json").

- 4.2. Listar las etiquetas "tag" (no numéricas) de los conjuntos train y test.

## 4.3. Mostrar :

- La sentencia train[0] con sus etiquetas tag[0].

- La sentencia validation[0] con su etiqueta tag[0].

4.4. Tokenizar la sentencia y el etiquetado de la cadena más larga del conjunto de entrenamiento de BioBERT usando los tokenizadores de BETO,,,XML-Roberta-large y GPT2 [URL 🔗](https://huggingface.co/FacebookAI/xlm-roberta-large)

## Ejemplo de uso

## Ejemplo de uso conBETO, [URL 🔗](https://huggingface-co.translate.goog/dccuchile/bert-base-spanish-wwm-cased?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc)

Nota: Importante que las etiquetas impresas no sean numéricas.

- 5. Clasificación de texto usando Regresión Logística y los LLMs de BETO, XLM-RoBERTa


En este ejercicio se aplicarán los conceptos de Machine Learning y Transfer Learning para la clasificación de sentimientos sobre el dataset Sarcasmo (local): [URL 🔗](https://drive.google.com/drive/u/0/folders/1PzGiow3fFXeX4WawhzB2rOyXalPmZte6)

## 5.1 MACHINE LEARNING

Dado un texto en español, ¿es sarcástico o no?. El sarcasmo es difícil de detectar automáticamente porque la ironía no está en las palabras individuales sino en la intención detrás de ellas. Una frase como “¡Qué maravilla, llegaste tarde otra vez!” usa palabras positivas pero comunica algo negativo.

La idea es la de entrenar un clasificador binario supervisado basado en una Regresión Logística que funcione como un pipeline clásico de Machine Learning, para ello se deberá tener en cuenta:

- 1. Datos — Dataset Sarcasmo, con textos etiquetados manualmente como sarcástico (1) o no sarcástico (0). [URL 🔗](https://drive.google.com/drive/u/0/folders/1PzGiow3fFXeX4WawhzB2rOyXalPmZte6)

- 2. Representación del texto (TF-IDF) — Convierte cada texto en un vector numérico contando qué palabras y bigramas aparecen, ponderando por su rareza en el corpus (sublinear_tf, max_features=50 000). No captura semántica profunda, pero es rápido e interpretable.

- 3. Clasificador (Regresión Logística) — Aprende qué combinaciones de n-gramas predicen sarcasmo. Usa class_weight="balanced" porque los datos pueden estar desbalanceados entre clases.

- 4. Evaluación — Métricas F1 (macro y ponderado) + matriz de confusión sobre validación.

- 5. Interfaz Streamlit con tres pestañas:

- Probar texto individual o en lote

- Ver métricas y matriz de confusión

- Explorar los n-gramas más asociados a cada clase (interpretabilidad del modelo)

Limitación central del enfoque: TF-IDF + LogReg captura patrones léxicos (“otra vez”, “qué maravilla”) pero no entiende contexto ni tono. Un modelo basado en embeddings (BERT, RoBERTa) sería más preciso pero más costoso de entrenar e inferir.


A continuación un diagrama de flujo del clasificador binario.


01 - DATOS

*5.2 TRANSFER LEARNING. Se deben afinar y entrenar el dastase Sarcasmo (anotado con dos clases: sarcasmo y no_sarcasmo). Para ello se utilizarán los modelos de clasificación BETO y XML-Roberta-large. Implementar el ajuste o fine tuning para los modelos BETO y XML-Roberta-large sobre el dataset Sarcasmo considerando todas las fases; preprocesamiento, alineación del dataset, tokenización, padding (mapeo) y el entrenamiento. Como insumo, se comparte el F1-score con batch size de 16 del modelo BETO sobre Sarcasmo [URL 🔗](https://huggingface-co.translate.goog/FacebookAI/xlm-roberta-large?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc)*


En esta misma dirección, es importante probar varias combinaciones de batch size según la capacidad de la GPU (8, 16 y 32 GB), utilizando GPU de Colab o Kaggle para ambos afinamientos. En el caso de XML-Roberta-large con 32 GB puede ser que no funcione por lo grande la parametrización + el dataset y por lo tanto hay que definir la cuantización para disminuir la capacidad de la GPU de Colab (Una buena alternativa es Kaggle ). [URL 🔗](https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll02-spanish)

- 3. Implementar las métricas de desempeño (f1-score) para batch size de 8, 16 y 32 ejecutando entre 6 y 10 épocas con Early Stopping. El modelo entrenado se deberá subir como un pipeline en Hugging Face [URL 🔗](https://huggingface.co/)

- 4. Los modelos se deben publicarse en Hugging face y utilizarse para clasificar las siguientes sentencias:

Para probar los modelos entrenados con el dataset Sarcasmo, usar los siguientes ejemplos. [URL 🔗](https://drive.google.com/drive/u/0/folders/1PzGiow3fFXeX4WawhzB2rOyXalPmZte6)

Texto: ¡Qué divertido, otro lunes de trabajo! 🙃

Predicción: SARCASMO

Confianza: 100.00%

Prob. sarcasmo: 100.00%

Texto: Me encanta cuando llueve el día que tengo planes al aire libre

Predicción: NO_SARCASMO


Confianza: 99.86%

Prob. sarcasmo: 0.14%

Texto: Obvio que aprobé el examen, estudié exactamente 5 minutos

Predicción: SARCASMO

Confianza: 100.00%

Prob. sarcasmo: 100.00%

Texto: Qué sorpresa que el internet se cayó justo cuando iba a entregar la tarea

Predicción: SARCASMO

Confianza: 100.00%

Prob. sarcasmo: 100.00%

Texto: Sí, porque claramente yo soy el experto en cocina gourmet 👨🍳

Predicción: SARCASMO

Confianza: 100.00%

Prob. sarcasmo: 100.00%

## 5. Realizar un análisis completo respondiendo:

¿Por qué si el tema del sarcasmo es tan complejo porque el rendimiento de BETO Sarcasmo es del 92% y 93% en XML-Roberta-large [URL 🔗](https://huggingface-co.translate.goog/FacebookAI/xlm-roberta-large?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc)

¿Por qué el rendimiento de la regresión lineal (85.8%) es menor que los dos modelos de transformers?

¿Cuál es la diferencia de usar TF-IDF en la regresión lineal y de usar el modelo de atención en los dos transformer?

¿Cuál es el efecto de la variación y parametrización del batch size en los modelos de transformers?

¿Son suficientes la cantidad de datos del dataset Sarcasmo para un problema tan complejo? [URL 🔗](https://drive.google.com/drive/u/0/folders/1PzGiow3fFXeX4WawhzB2rOyXalPmZte6)

Crear un conjunto de test de 30 sentencias (20 sarcásticas y 10 No) la idea es que se dividan en sentencias simples (10 palabras), medianas (25 palabras) y complejas (mayores de 40 palabras). Obtener el desempeño de cada uno de los tres clasificadores con respecto al conjunto de testeo.

Nota: La nota del taller se reparte de la siguiente manera; exposición 60% y 40% la entrega del informe del taller con explicaciones, tablas, resultados y análisis. Es importante que los modelos que se suban a Hugging Face sean los mejores y también con toda la información del entrenamiento.
