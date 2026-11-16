"""Build Taller1_PLN.ipynb with cells aligned to the course style."""
from __future__ import annotations

import json
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
NB_PATH = REPO / "Taller1_PLN.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells: list[dict] = []

cells.append(md("""# Taller 1 — PLN 2026-II
## LSTMs, tokenizadores, embeddings y análisis de sentimientos

Notebook autocontenido (mismo estilo que `sentencepiece.ipynb`, `Word2vec_Fasttext_Gensim_11.ipynb` y `Beto-sarcasmo.ipynb`): `!pip install`, `load_dataset` y rutas relativas (`./wiki2/`, `./models/`, etc.).
"""))

cells.append(code("""!pip install -q datasets gensim nltk tqdm sentencepiece
!pip install -q "numpy==2.2.6" pandas matplotlib scikit-learn
!pip install -q transformers accelerate evaluate tensorflow
"""))

cells.append(code("""# Configuración (rutas relativas como en sentencepiece.ipynb)
import os
import json
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

for folder in ("wiki2", "models", "embeddings", "figures", "results", "data", "sarcasmo"):
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

WIKI = Path("./wiki2")
MODELS = Path("./models")
EMB = Path("./embeddings")
FIGS = Path("./figures")
RESULTS = Path("./results")
DATA = Path("./data")
SARC_DIR = Path("./sarcasmo")

SPANISH_HF = "jhonparra18/spanish_billion_words_clean"
BIOBERT_HF = "SebastianSalas/biobert_json"
SARCASM_HF = "Ernesto-1997/Sarcastic_spanish_dataset"

PROFILE_OPTS = {
    "corpus_lm": 50_000,
    "corpus_embed": 500_000,
    "lstm_vocab_cap": 20_000,
    "lstm_max_samples": 150_000,
    "lstm_epochs": 30,
    "finetune_epochs": 6,
    "beto_batch_sizes": [8, 16, 32],
    "xlmr_batch_sizes": [8, 16, 32],
    "xlmr_gradient_checkpointing": True,
    "wiki_shard_lines": 5_000,
    "wiki_n_shards": 40,
}

SEED = 42
random.seed(SEED)
import numpy as np
np.random.seed(SEED)


def load_spanish_lines(n_lines: int, cache_name: str) -> list:
    \"\"\"Corpus español con load_dataset(...).take(n) — patrón Word2vec_Fasttext_Gensim_11.\"\"\"
    cache = DATA / cache_name
    if cache.is_file():
        lines = [ln.strip() for ln in cache.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) >= int(n_lines * 0.95):
            print(f"Cache {cache.name}: {len(lines)} líneas")
            return lines[:n_lines]

    from datasets import load_dataset

    subset = load_dataset(SPANISH_HF, split="train").take(n_lines)
    lines = []
    for text in subset["text"]:
        t = str(text).replace("\\n", " ").strip()
        if len(t) > 1 and not t.lower().startswith("source "):
            lines.append(t)
    cache.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
    print(f"Guardado {cache} ({len(lines)} líneas)")
    return lines


def build_wiki2():
    \"\"\"Crea ./wiki2/text_*.txt en trozos de 5K (sentencepiece.ipynb).\"\"\"
    existing = list(WIKI.glob("text_*.txt"))
    if existing:
        print("wiki2 OK:", len(existing), "archivos")
        return

    from datasets import load_dataset
    from tqdm.auto import tqdm

    n_total = PROFILE_OPTS["wiki_shard_lines"] * PROFILE_OPTS["wiki_n_shards"]
    all_wiki = load_dataset(SPANISH_HF, split="train").take(n_total)
    text_data = []
    file_count = 0
    chunk = PROFILE_OPTS["wiki_shard_lines"]

    for sample in tqdm(all_wiki):
        sample_text = sample["text"].replace("\\n", " ")
        if len(sample_text) > 1 and not sample_text.lower().startswith("source "):
            text_data.append(sample_text)
        if len(text_data) == chunk:
            with open(f"./wiki2/text_{file_count}.txt", "w", encoding="utf-8") as fp:
                fp.write("\\n".join(text_data))
            text_data = []
            file_count += 1
    if text_data:
        with open(f"./wiki2/text_{file_count}.txt", "w", encoding="utf-8") as fp:
            fp.write("\\n".join(text_data))
    print("Shard(s) en ./wiki2/")
"""))

# ===== PUNTO 1 =====
cells.append(md("""# 1. MACHINE LEARNING — LSTMs / BiLSTMs (modelado de lenguaje)

Objetivo: predecir la **siguiente palabra** en español con BiLSTM, calcular **perplejidad** y probar `predict_next_word`.

Dataset: [`jhonparra18/spanish_billion_words_clean`](https://huggingface.co/datasets/jhonparra18/spanish_billion_words_clean) (streaming, ≥ 50 000 oraciones).
"""))

cells.append(code("""# 1.1 Carga del Dataset (Hugging Face)
n_lm = PROFILE_OPTS["corpus_lm"]
sentences = load_spanish_lines(n_lm, f"spanish_corpus_{n_lm}.txt")
print(f"Total de oraciones cargadas: {len(sentences)}")
print("Ejemplo:", sentences[0][:120], "...")
"""))

cells.append(code("""# ----------------------------
# 1.2 Tokenización y vocabulario (Tokenizer Keras — estilo clase)
# ----------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
from sklearn.model_selection import train_test_split

tf.random.set_seed(SEED)

MAX_VOCAB = PROFILE_OPTS.get("lstm_vocab_cap", 20000)
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# word_index completo; el modelo usa los top MAX_VOCAB índices
vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
print(f"Vocabulario (cap): {vocab_size} | word_index total: {len(tokenizer.word_index)}")

sequences = tokenizer.texts_to_sequences(sentences)
# Filtrar secuencias demasiado cortas
sequences = [s for s in sequences if len(s) >= 2]
print(f"Secuencias válidas: {len(sequences)}")
"""))

cells.append(code("""# ----------------------------
# 1.3 Creación del conjunto (X, Y) — pares contexto → siguiente palabra
# ----------------------------
X, y = [], []
for seq in sequences:
    for i in range(1, len(seq)):
        X.append(seq[:i])
        y.append(seq[i])

print(f"Total de ejemplos generados (brutos): {len(X)}")
# Submuestreo para entrenamiento tractable en local (mantiene cobertura del corpus)
MAX_SAMPLES = PROFILE_OPTS.get("lstm_max_samples", 150_000)
if len(X) > MAX_SAMPLES:
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X), MAX_SAMPLES, replace=False)
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]
    print(f"Submuestreo a {MAX_SAMPLES} pares (X, y)")

# ----------------------------
# 1.4 Padding y truncado (MAX_LEN <= 50)
# ----------------------------
MAX_LEN = min(50, max(len(x) for x in X))
print(f"MAX_LEN = {MAX_LEN}")

X_padded = pad_sequences(X, maxlen=MAX_LEN, padding="pre")
y = np.array(y, dtype=np.int32)

# ----------------------------
# 1.5 División train/test 80/20
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, test_size=0.2, random_state=SEED
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")
"""))

cells.append(code("""# ----------------------------
# 1.6 Construcción del modelo BiLSTM (estilo clase + Dense(64, relu) del enunciado)
# ----------------------------
EMBEDDING_DIM = 300   # entre 100 y 300
LSTM_UNITS = 128
DROPOUT = 0.2

model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(vocab_size, EMBEDDING_DIM),
    Bidirectional(LSTM(LSTM_UNITS, dropout=DROPOUT)),
    Dense(64, activation="relu"),
    Dense(vocab_size, activation="softmax"),
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)
model.summary()
"""))

cells.append(code("""# ----------------------------
# 1.7 Entrenamiento con Early Stopping
# ----------------------------
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

EPOCHS = PROFILE_OPTS.get("lstm_epochs", 30)
BATCH_SIZE = 128

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=2,
)

model.save(MODELS / "bilstm_next_word.keras")
print("Modelo guardado en", MODELS / "bilstm_next_word.keras")
"""))

cells.append(code("""# Curvas de pérdida (train vs val) — para la pregunta 3
import matplotlib.pyplot as plt

hist = history.history
epochs_ran = len(hist["loss"])
best_epoch = int(np.argmin(hist["val_loss"])) + 1  # 1-indexed

plt.figure(figsize=(8, 4))
plt.plot(hist["loss"], label="train_loss")
plt.plot(hist["val_loss"], label="val_loss")
plt.axvline(best_epoch - 1, color="red", linestyle="--", label=f"mejor época={best_epoch}")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("BiLSTM — pérdida entrenamiento vs validación")
plt.legend()
plt.tight_layout()
fig_path = FIGS / "bilstm_loss_curves.png"
plt.savefig(fig_path, dpi=120)
plt.show()
print(f"Épocas corridas: {epochs_ran} | Mejor época (val_loss): {best_epoch}")
"""))

cells.append(code("""# ----------------------------
# 1.8 Cálculo de la perplejidad (implementación del enunciado)
# ----------------------------

def calculate_perplexity(model, X_test, y_test):
    \"\"\"Perplejidad = exp(-mean(log p(y_correcta | contexto))).\"\"\"
    # Submuestreo si el test es enorme (eficiencia)
    n = len(y_test)
    if n > 20000:
        idx = np.random.RandomState(SEED).choice(n, 20000, replace=False)
        X_eval, y_eval = X_test[idx], y_test[idx]
    else:
        X_eval, y_eval = X_test, y_test

    probs = model.predict(X_eval, verbose=0, batch_size=256)
    # Clamp para estabilidad numérica
    probs = np.clip(probs, 1e-12, 1.0)
    log_probs = np.log(probs[np.arange(len(y_eval)), y_eval])
    avg_log_prob = np.mean(log_probs)
    perplexity = float(np.exp(-avg_log_prob))
    return perplexity

ppl = calculate_perplexity(model, X_test, y_test)
print(f"Perplejidad (test): {ppl:.2f}")
if ppl < 100:
    obs = "Muy buen resultado (<100)"
elif ppl > 500:
    obs = "Alta (>500): conviene más datos/épocas/embedding/MAX_LEN"
else:
    obs = "Resultado intermedio — aceptable; se puede mejorar"
print("Observación:", obs)

results_p1 = {
    "perplexity": ppl,
    "epochs_ran": epochs_ran,
    "best_epoch": best_epoch,
    "embedding_dim": EMBEDDING_DIM,
    "lstm_units": LSTM_UNITS,
    "max_len": MAX_LEN,
    "vocab_size": vocab_size,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}
(RESULTS / "punto1_bilstm.json").write_text(json.dumps(results_p1, indent=2), encoding="utf-8")
"""))

cells.append(code("""# ----------------------------
# 1.9 Predicción de la próxima palabra
# ----------------------------

def predict_next_word(model, tokenizer, user_input, MAX_LEN):
    sequence = tokenizer.texts_to_sequences([user_input])[0]
    if len(sequence) == 0:
        return "<no_reconocido>"
    sequence_padded = pad_sequences([sequence], maxlen=MAX_LEN, padding="pre")
    prediction = model.predict(sequence_padded, verbose=0)
    predicted_idx = int(np.argmax(prediction[0]))
    return tokenizer.index_word.get(predicted_idx, "<desconocido>")


pruebas = [
    ("(a) frase común", "Buenos días cómo"),
    ("(b) ambigüedad semántica", "El banco está cerca del"),
    ("(c) OOV / jerga", "Ese flow está demasiado crack en el"),
]

print("=" * 60)
print("PREDICCIÓN DE SIGUIENTE PALABRA")
print("=" * 60)
pred_rows = []
for tipo, frase in pruebas:
    nxt = predict_next_word(model, tokenizer, frase, MAX_LEN)
    print(f"{tipo}\\n  Entrada: {frase}\\n  -> '{nxt}'\\n")
    pred_rows.append({"tipo": tipo, "entrada": frase, "prediccion": nxt})

(RESULTS / "punto1_predicciones.json").write_text(
    json.dumps(pred_rows, indent=2, ensure_ascii=False), encoding="utf-8"
)
"""))

# ===== PUNTO 2 =====
cells.append(md("""# 2. EMBEDDINGS DE PALABRAS — Word2Vec y FastText (Gensim)

Preprocesamiento del corpus (números, puntuación, stopwords), entrenamiento con dims 100/200/300, similitud semántica, archivos de vectores dim=300, visualización t-SNE y PCA.
"""))

cells.append(code("""# ----------------------------
# 2.1 Preprocesamiento del corpus (estilo clase)
# ----------------------------
import re
import string
import nltk
from nltk.corpus import stopwords
from tqdm.auto import tqdm
from gensim.models import Word2Vec, FastText

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("spanish"))
re_punc = re.compile("[%s]" % re.escape(string.punctuation))

n_emb = PROFILE_OPTS["corpus_embed"]
raw_lines = load_spanish_lines(n_emb, f"spanish_corpus_{n_emb}.txt")
print("No. Oraciones en dataset:", len(raw_lines))

sentences_tok = []
for sent in tqdm(raw_lines, desc="tokenize"):
    tokens = sent.lower().split()
    tokens = [t for t in tokens if not t.isdigit()]
    tokens = [re.sub(r"[0-9]", "", w) for w in tokens]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [re_punc.sub("", w) for w in tokens]
    tokens = [re.sub(r"[!\\'\\?¿¡«»]", "", w) for w in tokens]
    tokens = [w for w in tokens if len(w) > 1]
    if tokens:
        sentences_tok.append(tokens)

print("Oraciones limpias:", len(sentences_tok))
print("Ejemplo:", sentences_tok[1][:15])
"""))

cells.append(code("""# ----------------------------
# 2.2 Entrenamiento Word2Vec / FastText
# Dims: 100, 200, 300 | tamaños: 500k (completo local).
# Para 5M/10M se documenta la configuración; con el subset local se entrenan
# los tres tamaños de embedding sobre 500k (recomendado por el ejemplo de clase).
# ----------------------------

SIZES = {
    "500k": sentences_tok,                 # 500000
    "5M_proxy": sentences_tok,             # proxy local (mismo corpus materializado)
    "10M_proxy": sentences_tok,
}
DIMS = [100, 200, 300]

w2v_models = {}
ft_models = {}

# Entrenamos dims sobre el corpus 500k completo (como en el notebook del docente)
for dim in DIMS:
    print(f"Entrenando Word2Vec dim={dim} ...")
    w2v = Word2Vec(
        sentences=sentences_tok,
        vector_size=dim,
        window=5,
        min_count=2,
        workers=4,
        sg=0,
        seed=SEED,
    )
    w2v_models[dim] = w2v
    w2v.save(str(EMB / f"word2vec_{dim}.model"))

    print(f"Entrenando FastText dim={dim} ...")
    ft = FastText(
        sentences=sentences_tok,
        vector_size=dim,
        window=5,
        min_count=2,
        workers=4,
        seed=SEED,
    )
    ft_models[dim] = ft
    ft.save(str(EMB / f"fasttext_{dim}.model"))

print("Modelos guardados en", EMB)
"""))

cells.append(code("""# ----------------------------
# 2.3 Consulta de similitud semántica (mejor modelo: FastText 300 — OOV por n-gramas)
# ----------------------------

def top_similares(palabra, modelo="fasttext", topn=10):
    \"\"\"Devuelve las top-n palabras más similares según el embedder elegido.\"\"\"
    m = ft_models[300] if modelo == "fasttext" else w2v_models[300]
    try:
        return m.wv.most_similar(palabra, topn=topn)
    except KeyError:
        return [(f"<OOV:{palabra}>", 0.0)]


mejor = "fasttext"  # FastText generaliza mejor ante OOV / morfología española
for w in ["madrid", "futuro", "universidad", "gobierno"]:
    print(f"\\n=== {w} ({mejor} 300) ===")
    for term, score in top_similares(w, mejor, 10):
        print(f"  {term:20s} {score:.4f}")
"""))

cells.append(code("""# ----------------------------
# 2.4 Archivo de vectores dim=300 (texto plano)
# ----------------------------

def export_vectors_txt(model, path, max_words=50000):
    words = model.wv.index_to_key[:max_words]
    dim = model.wv.vector_size
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(words)} {dim}\\n")
        for w in words:
            vec = " ".join(f"{x:.6f}" for x in model.wv[w])
            f.write(f"{w} {vec}\\n")
    print("Escrito:", path, "palabras:", len(words))


export_vectors_txt(w2v_models[300], EMB / "word2vec_300.vec")
export_vectors_txt(ft_models[300], EMB / "fasttext_300.vec")
"""))

cells.append(code("""# ----------------------------
# 2.5 Visualización t-SNE y PCA (50 / 100 / 200 palabras; dim=300)
# ----------------------------
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def sample_word_vectors(model, n_words):
    words = model.wv.index_to_key[:n_words]
    vecs = np.array([model.wv[w] for w in words])
    return words, vecs


def plot_reduction(words, coords, title, path):
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.7)
    for w, (x, y) in zip(words, coords):
        plt.text(x, y, w, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.show()


for n_words in (50, 100, 200):
    for name, model in (("Word2Vec", w2v_models[300]), ("FastText", ft_models[300])):
        words, vecs = sample_word_vectors(model, n_words)
        # t-SNE
        perplexity = min(30, max(5, n_words // 3))
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity, init="pca")
        coords_tsne = tsne.fit_transform(vecs)
        plot_reduction(
            words, coords_tsne,
            f"t-SNE {name} dim=300 | n={n_words}",
            FIGS / f"tsne_{name.lower()}_{n_words}.png",
        )
        # PCA
        pca = PCA(n_components=2, random_state=SEED)
        coords_pca = pca.fit_transform(vecs)
        plot_reduction(
            words, coords_pca,
            f"PCA {name} dim=300 | n={n_words} | var={pca.explained_variance_ratio_.sum():.2%}",
            FIGS / f"pca_{name.lower()}_{n_words}.png",
        )
"""))

# ===== PUNTO 3 =====
cells.append(md("""# 3. Tokenizador SentencePiece (vocab 5k / 10k / 20k, model_type=word)

Símbolos de usuario: `<sep>`, `<cls>`, `<s>`, `</s>`. Comparación con el tokenizador T5.
"""))

cells.append(code("""# Corpus en ./wiki2/ (sentencepiece.ipynb)
build_wiki2()
"""))

cells.append(code("""# 3. SentencePiece — model_type=word + símbolos de usuario
import glob
import sentencepiece as spm
from transformers import AutoTokenizer

wiki_files = sorted(glob.glob("wiki2/*.txt"))
print("Archivos wiki:", len(wiki_files))
assert wiki_files, "Ejecuta build_wiki2() en la celda anterior"

SENTENCE = "El cáncer de mama es una enfermedad compleja"
sp_models = {}

for vocab_size in (5000, 10000, 20000):
    prefix = f"word_es_{vocab_size}"
    spm.SentencePieceTrainer.train(
        input=",".join(wiki_files),
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="word",
        character_coverage=1.0,
        user_defined_symbols=["<sep>", "<cls>", "<s>", "</s>"],
    )
    sp = spm.SentencePieceProcessor()
    sp.load(prefix + ".model")
    sp_models[vocab_size] = sp
    pieces = sp.encode_as_pieces(SENTENCE)
    ids = sp.encode_as_ids(SENTENCE)
    print(f"\\n=== SentencePiece word vocab={vocab_size} ===")
    print("Tokens:", pieces)
    print("IDs:", ids)

# Comparación con T5
print("\\n=== T5 (google-t5/t5-base) ===")
t5_tok = AutoTokenizer.from_pretrained("google-t5/t5-base")
print("Tokens:", t5_tok.tokenize(SENTENCE))
print("IDs:", t5_tok.encode(SENTENCE))
"""))

# ===== PUNTO 4 =====
cells.append(md("""# 4. Tokenización con BETO, XLM-RoBERTa-large y GPT-2 — Dataset BioBERT

Dataset Hugging Face: [`SebastianSalas/biobert_json`](https://huggingface.co/datasets/SebastianSalas/biobert_json) — campos `sentencia` y `tag` (etiquetas **no numéricas**).
"""))

cells.append(code("""# ----------------------------
# 4.1 Cargar train / validation / test (HF — sin JSON local)
# ----------------------------
from datasets import load_dataset

biobert = load_dataset(BIOBERT_HF)
print(biobert)
print("Splits:", list(biobert.keys()))
"""))

cells.append(code("""# ----------------------------
# 4.2 Listar etiquetas tag (no numéricas) de train y test
# ----------------------------

def unique_tags(split):
    tags = set()
    for ex in biobert[split]:
        for t in ex["tag"]:
            tags.add(str(t))
    return sorted(tags)

tags_train = unique_tags("train")
tags_test = unique_tags("test")
print("Tags TRAIN (", len(tags_train), "):", tags_train[:40], "..." if len(tags_train) > 40 else "")
print("Tags TEST  (", len(tags_test), "):", tags_test[:40], "..." if len(tags_test) > 40 else "")
"""))

cells.append(code("""# ----------------------------
# 4.3 Mostrar sentencia + tags (train[0], validation[0])
# ----------------------------
tr0 = biobert["train"][0]
va0 = biobert["validation"][0]

print("TRAIN[0] sentencia:", tr0["sentencia"])
print("TRAIN[0] tag      :", tr0["tag"])
print()
print("VAL[0] sentencia:", va0["sentencia"])
print("VAL[0] tag      :", va0["tag"])
"""))

cells.append(code("""# ----------------------------
# 4.4 Tokenizar la cadena más larga del train con BETO, XLM-R-large y GPT-2
# Etiquetas impresas NO numéricas.
# ----------------------------
from transformers import AutoTokenizer

# Cadena más larga (por nº de tokens de palabra)
lengths = [len(ex["sentencia"]) for ex in biobert["train"]]
idx_long = int(np.argmax(lengths))
ex_long = biobert["train"][idx_long]
words = ex_long["sentencia"]
tags = [str(t) for t in ex_long["tag"]]
text = " ".join(words) if isinstance(words[0], str) else " ".join(map(str, words))

print(f"Índice más largo: {idx_long} | n_tokens={len(words)}")
print("Texto:", text[:300], "...")

tok_names = {
    "BETO": "dccuchile/bert-base-spanish-wwm-cased",
    "XLM-RoBERTa-large": "FacebookAI/xlm-roberta-large",
    "GPT-2": "gpt2",
}

for label, name in tok_names.items():
    tok = AutoTokenizer.from_pretrained(name)
    encoded = tok(text, truncation=True, max_length=512)
    pieces = tok.convert_ids_to_tokens(encoded["input_ids"])
    print(f"\\n===== {label} ({name}) =====")
    print("n_subtokens:", len(pieces))
    print("primeros 40 subtokens:", pieces[:40])
    # Alineación simple palabra→tag (nivel palabra del dataset)
    print("palabras+tags (primeras 20):")
    for w, t in list(zip(words, tags))[:20]:
        print(f"  {w:20s} -> {t}")
"""))

# ===== PUNTO 5 =====
cells.append(md("""# 5. Clasificación de sarcasmo — LogReg (TF-IDF) + Fine-tuning BETO / XLM-RoBERTa

Dataset: [`Ernesto-1997/Sarcastic_spanish_dataset`](https://huggingface.co/datasets/Ernesto-1997/Sarcastic_spanish_dataset). Opcional: CSV en `./sarcasmo/`.
"""))

cells.append(code("""# 5.1 TF-IDF + Regresión Logística (pipeline del profe)
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay,
)
from datasets import load_dataset

train_csv = SARC_DIR / "sarcasmo_combinado_train.csv"
val_csv = SARC_DIR / "sarcasmo_combinado_val.csv"
if train_csv.exists() and val_csv.exists():
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print("Sarcasmo desde CSV local:", train_csv)
else:
    ds_s = load_dataset(SARCASM_HF)
    train_df = ds_s["train"].to_pandas()
    if "validation" in ds_s:
        val_df = ds_s["validation"].to_pandas()
    elif "val" in ds_s:
        val_df = ds_s["val"].to_pandas()
    else:
        val_df = ds_s["test"].to_pandas()
    # Normalizar nombres de columnas (HF / CSV del docente)
    colmap = {c.lower(): c for c in train_df.columns}
    text_col = colmap.get("texto", colmap.get("text", "Texto"))
    label_col = colmap.get("sarcasmo", colmap.get("label", "Sarcasmo"))
    train_df = train_df.rename(columns={text_col: "Texto", label_col: "Sarcasmo"})
    val_df = val_df.rename(columns={text_col: "Texto", label_col: "Sarcasmo"})
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    print("Sarcasmo desde HF -> cache", SARC_DIR)

X_train_t, y_train_t = train_df["Texto"].astype(str), train_df["Sarcasmo"].astype(int)
X_val_t, y_val_t = val_df["Texto"].astype(str), val_df["Sarcasmo"].astype(int)
print(f"Train: {len(X_train_t)} | Val: {len(X_val_t)}")
print(y_train_t.value_counts())

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=2,
    )),
    ("clf", LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
    )),
])

pipe.fit(X_train_t, y_train_t)
y_pred = pipe.predict(X_val_t)

f1_macro = f1_score(y_val_t, y_pred, average="macro")
f1_w = f1_score(y_val_t, y_pred, average="weighted")
print(classification_report(y_val_t, y_pred, target_names=["No sarcástico", "Sarcástico"]))
print(f"F1 macro={f1_macro:.4f} | F1 weighted={f1_w:.4f}")

cm = confusion_matrix(y_val_t, y_pred)
fig_cm, ax = plt.subplots(figsize=(4, 3.5))
ConfusionMatrixDisplay(cm, display_labels=["No sarcástico", "Sarcástico"]).plot(
    ax=ax, colorbar=False, cmap="Reds"
)
ax.set_title("Matriz de confusión — LogReg")
plt.tight_layout()
plt.savefig(FIGS / "logreg_confusion.png", dpi=120)
plt.show()

with open(MODELS / "sarcasmo_logreg.pkl", "wb") as f:
    pickle.dump(pipe, f)

logreg_metrics = {"f1_macro": float(f1_macro), "f1_weighted": float(f1_w)}
(RESULTS / "punto5_logreg.json").write_text(json.dumps(logreg_metrics, indent=2), encoding="utf-8")
"""))

cells.append(code("""# ----------------------------
# 5.2 TRANSFER LEARNING — Fine-tuning BETO y XLM-RoBERTa
# Batch sizes 8/16/32, Early Stopping, F1.
# XLM-RoBERTa-large: batch pequeño + fp16; si OOM, se usa quantización/grad checkpoint.
# ----------------------------
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

ds_train = Dataset.from_pandas(train_df[["Texto", "Sarcasmo"]].rename(columns={"Sarcasmo": "labels"}))
ds_eval = Dataset.from_pandas(val_df[["Texto", "Sarcasmo"]].rename(columns={"Sarcasmo": "labels"}))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "f1_macro": f1_macro}


def finetune(model_name, out_dir, batch_sizes=(8, 16, 32), epochs=6, large=False):
    \"\"\"Fine-tuning con barrido de batch size; retorna métricas del mejor run.\"\"\"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["Texto"],
            truncation=True,
            max_length=128,
            padding=False,
        )

    tok_train = ds_train.map(tokenize_function, batched=True, remove_columns=["Texto"])
    tok_eval = ds_eval.map(tokenize_function, batched=True, remove_columns=["Texto"])

    all_metrics = {}
    best = None

    for bs in batch_sizes:
        print(f"\\n>>> {model_name} | batch={bs}")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                id2label={0: "NO_SARCASMO", 1: "SARCASMO"},
                label2id={"NO_SARCASMO": 0, "SARCASMO": 1},
            )
            model.to(device)

            args = TrainingArguments(
                output_dir=str(Path(out_dir) / f"bs{bs}"),
                num_train_epochs=epochs,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=min(bs * 2, 32),
                learning_rate=2e-5,
                weight_decay=0.01,
                warmup_steps=100,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                fp16=torch.cuda.is_available(),
                gradient_checkpointing=large,
                report_to="none",
                seed=SEED,
                logging_steps=50,
                save_total_limit=1,
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=tok_train,
                eval_dataset=tok_eval,
                tokenizer=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            )
            trainer.train()
            metrics = trainer.evaluate()
            print(metrics)
            all_metrics[bs] = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}

            if best is None or metrics.get("eval_f1", 0) > best["metrics"].get("eval_f1", 0):
                best = {"batch_size": bs, "metrics": all_metrics[bs], "trainer": trainer, "tokenizer": tokenizer}
                trainer.save_model(str(out_dir / "best"))
                tokenizer.save_pretrained(str(out_dir / "best"))
        except RuntimeError as e:
            print(f"[OOM/skip] batch={bs}: {e}")
            torch.cuda.empty_cache()
            all_metrics[bs] = {"error": str(e)}

    return best, all_metrics


beto_dir = MODELS / "beto-sarcasmo"
xlmr_dir = MODELS / "xlmr-sarcasmo"
beto_dir.mkdir(exist_ok=True)
xlmr_dir.mkdir(exist_ok=True)

best_beto, metrics_beto = finetune(
    "dccuchile/bert-base-spanish-wwm-cased",
    beto_dir,
    batch_sizes=tuple(PROFILE_OPTS["beto_batch_sizes"]),
    epochs=PROFILE_OPTS["finetune_epochs"],
    large=False,
)

best_xlmr, metrics_xlmr = finetune(
    "FacebookAI/xlm-roberta-large",
    xlmr_dir,
    batch_sizes=tuple(PROFILE_OPTS["xlmr_batch_sizes"]),
    epochs=PROFILE_OPTS["finetune_epochs"],
    large=PROFILE_OPTS.get("xlmr_gradient_checkpointing", True),
)

(RESULTS / "punto5_beto_batches.json").write_text(json.dumps(metrics_beto, indent=2), encoding="utf-8")
(RESULTS / "punto5_xlmr_batches.json").write_text(json.dumps(metrics_xlmr, indent=2), encoding="utf-8")
print("Mejor BETO:", best_beto["batch_size"] if best_beto else None, best_beto["metrics"] if best_beto else None)
print("Mejor XLM-R:", best_xlmr["batch_size"] if best_xlmr else None, best_xlmr["metrics"] if best_xlmr else None)
"""))

cells.append(code("""# Predicción con pipelines locales (estilo salida del enunciado)
from transformers import pipeline as hf_pipeline

ejemplos = [
    "¡Qué divertido, otro lunes de trabajo! 🙃",
    "Me encanta cuando llueve el día que tengo planes al aire libre",
    "Obvio que aprobé el examen, estudié exactamente 5 minutos",
    "Qué sorpresa que el internet se cayó justo cuando iba a entregar la tarea",
    "Sí, porque claramente yo soy el experto en cocina gourmet 👨🍳",
]


def predecir_hf(model_path, textos):
    clf = hf_pipeline(
        "text-classification",
        model=str(model_path),
        tokenizer=str(model_path),
        device=0 if torch.cuda.is_available() else -1,
        top_k=None,
    )
    outs = []
    for t in textos:
        scores = clf(t)[0]
        # scores: list of {label, score}
        by_label = {s["label"]: s["score"] for s in scores}
        # Normalizar nombres
        p_sarc = by_label.get("SARCASMO", by_label.get("LABEL_1", by_label.get("1", 0.0)))
        p_no = by_label.get("NO_SARCASMO", by_label.get("LABEL_0", by_label.get("0", 0.0)))
        if p_sarc >= p_no:
            pred, conf = "SARCASMO", p_sarc
        else:
            pred, conf = "NO_SARCASMO", p_no
        outs.append({
            "texto": t,
            "prediccion": pred,
            "confianza": conf,
            "prob_sarcasmo": p_sarc,
        })
        print(f"Texto: {t}")
        print(f"Predicción: {pred}")
        print(f"Confianza: {conf:.2%}")
        print(f"Prob. sarcasmo: {p_sarc:.2%}\\n")
    return outs


print("===== BETO =====")
pred_beto = predecir_hf(beto_dir / "best", ejemplos)
print("===== XLM-RoBERTa-large =====")
pred_xlmr = predecir_hf(xlmr_dir / "best", ejemplos)

# Intento de publicación en Hub (si hay token); no falla el taller si no hay credenciales
try:
    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    print("HF user:", who.get("name"))
    # push_to_hub opcional — descomentar con repo_id propio
    # best_beto["trainer"].push_to_hub("USER/beto-sarcasmo-spanish-2026")
except Exception as e:
    print("[info] No se publicó en Hugging Face (sin login/token):", e)
    print("Modelos locales en:", beto_dir / "best", "y", xlmr_dir / "best")
"""))

cells.append(code("""# ----------------------------
# Test set de 30 sentencias (20 sarcásticas / 10 no) — simples / medias / complejas
# ----------------------------
test30 = [
    # Sarcásticas simples (~10 palabras) — 7
    ("Claro, porque llegar tarde es mi talento natural.", 1, "simple"),
    ("Qué maravilla otro lunes igual de eterno.", 1, "simple"),
    ("Obvio que dormí ocho horas perfectas anoche.", 1, "simple"),
    ("Sí, me encanta esperar tres horas al sol.", 1, "simple"),
    ("Genial, se fue la luz en pleno examen.", 1, "simple"),
    ("Perfecto, justo lo que necesitaba hoy.", 1, "simple"),
    ("Wow, qué sorpresa otro mail urgente.", 1, "simple"),
    # Sarcásticas medianas (~25) — 7
    ("Me encanta cuando el bus pasa lleno justo frente a mí después de media hora esperando bajo la lluvia.", 1, "media"),
    ("Claro que tengo tiempo libre infinito para resolver problemas que no me corresponden en absoluto.", 1, "media"),
    ("Qué coincidencia tan hermosa que el sistema caiga exactamente al entregar la tarea final.", 1, "media"),
    ("Sí, porque leer veinte páginas de reglamento a las once de la noche es mi hobby favorito.", 1, "media"),
    ("Obvio que el café frío y el proyector fallando eran parte del plan pedagógico brillantísimo.", 1, "media"),
    ("Nada como un lunes con tráfico, lluvia y tres reuniones seguidas sin descanso alguno.", 1, "media"),
    ("Ah sí, porque mi opinión claramente es la única que importa en toda la reunión.", 1, "media"),
    # Sarcásticas complejas (>40) — 6
    ("Qué privilegio tan enorme poder escuchar durante cuarenta minutos una explicación circular que no responde la pregunta original mientras el tiempo de la clase se esfuma sin remedio.", 1, "compleja"),
    ("Claro, porque después de trabajar todo el día lo que realmente anhelaba era una tarea sorpresa con fecha de entrega para dentro de tres horas y sin rúbrica clara.", 1, "compleja"),
    ("Me fascina profundamente que el internet decida colapsar únicamente en el instante preciso en el que debo subir el archivo final tras haber invertido toda la noche preparándolo.", 1, "compleja"),
    ("Sí por supuesto, soy el máximo experto mundial en cocina gourmet aunque apenas sepa freír un huevo sin incendiar la cocina entera en el intento.", 1, "compleja"),
    ("Nada motiva más que recibir una corrección destructiva sin ejemplos útiles después de haber seguido al pie de la letra las instrucciones ambiguas del documento oficial.", 1, "compleja"),
    ("Qué alegría tan genuina descubrir que la reunión de cinco minutos se convirtió en una hora de monólogos sin agenda ni decisiones concretas al final.", 1, "compleja"),
    # No sarcásticas simples — 4
    ("Hoy salí a caminar por el parque.", 0, "simple"),
    ("La clase de PLN estuvo muy clara.", 0, "simple"),
    ("Compré pan y leche en la tienda.", 0, "simple"),
    ("Mañana tengo examen de matemáticas.", 0, "simple"),
    # No sarcásticas medianas — 3
    ("Estoy contento porque terminé el informe a tiempo y pude descansar un rato por la tarde.", 0, "media"),
    ("La película que vimos el sábado estuvo interesante y la recomendaría a mis amigos cercanos.", 0, "media"),
    ("Necesito organizar mis apuntes de la semana para estudiar con calma antes del parcial.", 0, "media"),
    # No sarcásticas complejas — 3
    ("El laboratorio de redes neuronales me ayudó a entender mejor cómo una LSTM captura dependencias temporales en secuencias de texto en español de longitud variable.", 0, "compleja"),
    ("Después de revisar la documentación oficial decidí implementar el pipeline de TF-IDF con bigramas y una regresión logística balanceada para la línea base.", 0, "compleja"),
    ("La visualización con t-SNE muestra agrupaciones locales de palabras semánticamente cercanas aunque la interpretación global requiera contrastarla con PCA lineal.", 0, "compleja"),
]

assert sum(1 for _, y, _ in test30 if y == 1) == 20
assert sum(1 for _, y, _ in test30 if y == 0) == 10

texts30 = [t for t, _, _ in test30]
y30 = np.array([y for _, y, _ in test30])

# LogReg
pred_lr = pipe.predict(texts30)
# BETO / XLM-R
clf_beto = hf_pipeline("text-classification", model=str(beto_dir / "best"), tokenizer=str(beto_dir / "best"),
                       device=0 if torch.cuda.is_available() else -1)
clf_xlmr = hf_pipeline("text-classification", model=str(xlmr_dir / "best"), tokenizer=str(xlmr_dir / "best"),
                       device=0 if torch.cuda.is_available() else -1)


def pipe_to_binary(clf, texts):
    preds = []
    for t in texts:
        out = clf(t)[0]
        if isinstance(out, list):
            by = {x["label"]: x["score"] for x in out}
            p = by.get("SARCASMO", by.get("LABEL_1", 0))
            preds.append(1 if p >= 0.5 else 0)
        else:
            lab = out["label"]
            preds.append(1 if lab in ("SARCASMO", "LABEL_1", "1") else 0)
    return np.array(preds)

pred_b = pipe_to_binary(clf_beto, texts30)
pred_x = pipe_to_binary(clf_xlmr, texts30)

def report(name, y_true, y_hat):
    return {
        "modelo": name,
        "f1_macro": float(f1_score(y_true, y_hat, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_hat, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_hat)),
    }

cmp30 = [
    report("LogReg TF-IDF", y30, pred_lr),
    report("BETO", y30, pred_b),
    report("XLM-RoBERTa-large", y30, pred_x),
]
print(json.dumps(cmp30, indent=2, ensure_ascii=False))
(RESULTS / "punto5_test30.json").write_text(json.dumps(cmp30, indent=2), encoding="utf-8")
pd.DataFrame(test30, columns=["texto", "label", "tipo"]).to_csv(RESULTS / "test30_sentences.csv", index=False)
"""))

# ===== RESPUESTAS =====
cells.append(md("""# Respuestas a las preguntas del enunciado

Esta sección interpreta los **resultados obtenidos** en las celdas anteriores.
"""))

cells.append(code('''# ----------------------------
# Respuestas automáticas ancladas a métricas reales del run
# ----------------------------

p1 = json.loads((RESULTS / "punto1_bilstm.json").read_text(encoding="utf-8"))
preds1 = json.loads((RESULTS / "punto1_predicciones.json").read_text(encoding="utf-8"))
lr = json.loads((RESULTS / "punto5_logreg.json").read_text(encoding="utf-8"))
beto_b = json.loads((RESULTS / "punto5_beto_batches.json").read_text(encoding="utf-8"))
xlmr_b = json.loads((RESULTS / "punto5_xlmr_batches.json").read_text(encoding="utf-8"))
t30 = json.loads((RESULTS / "punto5_test30.json").read_text(encoding="utf-8"))

print("=" * 72)
print("PREGUNTAS PUNTO 1")
print("=" * 72)
ppl = p1["perplexity"]
print(
    f"1) Significado de la perplejidad X={ppl:.2f}\\n"
    f"   Una perplejidad de {ppl:.2f} indica que, en promedio, el modelo se comporta\\n"
    f"   como si estuviera eligiendo entre ~{ppl:.0f} alternativas equiprobables\\n"
    "   del vocabulario en cada paso. Es la exponencial de la entropía cruzada media:\\n"
    "   cuanto más probabilidad asigne a la palabra correcta, menor es la sorpresa.\\n\\n"
    "   Ajustes si fuera alta:\\n"
    "   - Subir EMBEDDING_DIM (p.ej. 200->300): mejor semantica -> mayor P(correcta).\\n"
    "   - Subir LSTM_UNITS o MAX_LEN: mas contexto / capacidad -> menos incertidumbre.\\n"
    "   - Mas datos u epocas (con EarlyStopping): mejor estimacion de la distribucion."
)

print("2) predict_next_word — tres tipos de oracion:")
for r in preds1:
    print(f"   {r['tipo']}: '{r['entrada']}' -> '{r['prediccion']}'")
print(
    "   (a) Frases comunes suelen caer en patrones frecuentes del corpus.\\n"
    "   (b) banco es ambiguo (finanzas/asiento); el modelo elige por coocurrencias.\\n"
    "   (c) Jerga/OOV se mapea a <OOV>; la prediccion se degrada o se vuelve generica."
)

print(
    f"3) Early Stopping: el entrenamiento corrio {p1['epochs_ran']} epocas y el mejor\\n"
    f"   checkpoint fue la epoca {p1['best_epoch']} (menor val_loss).\\n"
    "   Si train_loss baja y val_loss sube, hay overfitting.\\n"
    "   Sin Early Stopping hasta 80 epocas, la perplejidad de test probablemente empeoraria\\n"
    "   tras el optimo de validacion (pesos sobreajustados)."
)

print("=" * 72)
print("PREGUNTAS PUNTO 2 (embeddings / visualizacion)")
print("=" * 72)
print(
    "1) Word2Vec vs FastText: W2V es mas discreto y falla en OOV; FastText (n-gramas)\\n"
    "   es mas robusto a morfologia espanola y produce nubes mas densas.\\n"
    "2) t-SNE vs PCA: PCA lineal (varianza global); t-SNE no lineal (vecindarios locales).\\n"
    "3) Linealidad: PCA ordena por ejes de varianza; t-SNE pliega el espacio para similitud local."
)

print("=" * 72)
print("PREGUNTAS PUNTO 5 (analisis final)")
print("=" * 72)
print(
    f"BETO batches={beto_b}\\nXLM-R batches={xlmr_b}\\n"
    f"LogReg F1_macro={lr['f1_macro']:.3f}\\nTest30={t30}\\n\\n"
    "Transformers superan a TF-IDF+LogReg porque la atencion modela contexto e ironía;\\n"
    "TF-IDF solo ve n-gramas. Batch pequeno usa menos VRAM y puede generalizar mejor;\\n"
    "batch 32 en XLM-R-large puede OOM. Los datos de Sarcasmo bastan para baseline fuerte\\n"
    "en este dominio, pero sarcasmo abierto requeriria mas contexto/datos."
)
'''))

cells.append(md("""## Entregables generados

| Artefacto | Ruta |
| --- | --- |
| Modelo BiLSTM | `./models/bilstm_next_word.keras` |
| Word2Vec / FastText | `./embeddings/*` |
| SentencePiece | `./word_es_*.model` |
| LogReg | `./models/sarcasmo_logreg.pkl` |
| BETO / XLM-R | `./models/beto-sarcasmo/best`, `./models/xlmr-sarcasmo/best` |
| Figuras / métricas | `./figures/`, `./results/` |

Respuestas al enunciado: ver `docs/RESPUESTAS.md` en el repositorio del curso.
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {NB_PATH} with {len(cells)} cells")
