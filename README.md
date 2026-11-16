# Taller 1 — PLN 2026-II · v1.0.0

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/version-1.0.0-blue)

LSTMs/BiLSTM, Word2Vec/FastText, SentencePiece, tokenización (BETO, XLM-R, GPT-2) y clasificación de sarcasmo (TF-IDF + fine-tuning).

| Recurso | Ubicación |
| --- | --- |
| **Notebook (entrega)** | [`Taller1_PLN.ipynb`](Taller1_PLN.ipynb) |
| Respuestas al enunciado | [`docs/RESPUESTAS.md`](docs/RESPUESTAS.md) |
| Enunciado + ejemplos del docente | [`base/`](base/) |

## Uso del notebook

1. Abre o sube solo **`Taller1_PLN.ipynb`** y ejecuta las celdas en orden.
2. La primera celda instala dependencias con `!pip install` (igual que en `base/help/`).
3. Los datos se cargan desde Hugging Face; los archivos generados quedan en el directorio de trabajo (`./wiki2/`, `./models/`, etc.).

## Ejecución local (opcional)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook Taller1_PLN.ipynb
```

## Estructura

```
.
├── Taller1_PLN.ipynb
├── docs/RESPUESTAS.md
├── base/                 # Enunciado y notebooks de referencia
├── requirements.txt
└── scripts/build_notebook.py   # Regenerar el ipynb (mantenimiento)
```

Repositorio: [AndreyQuicenoC/Taller_1-NLP](https://github.com/AndreyQuicenoC/Taller_1-NLP)

Universidad del Valle — PLN 2026-II.