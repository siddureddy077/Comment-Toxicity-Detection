# Comment-Toxicity-Detection


A deep learning-based system that detects toxic comments (toxic, severe_toxic,
obscene, threat, insult, identity_hate) using a Bidirectional LSTM neural
network, served through an interactive Streamlit web application.

## Problem Statement

Online communities face challenges moderating harassment, hate speech, and
offensive language at scale. This project builds a deep learning model that
analyzes comment text and predicts the probability of toxicity across six
categories, enabling real-time flagging and bulk moderation via CSV upload.

## Project Structure

```
toxicity_project/
├── data/
│   ├── train.csv              # Jigsaw toxic comment training data (159,571 rows)
│   └── test.csv                # Unlabeled test set (153,164 rows)
├── models/
│   ├── toxicity_model.keras    # Trained Bidirectional LSTM model
│   ├── tokenizer.pkl           # Fitted Keras tokenizer
│   ├── config.json             # Model config (vocab size, max_len, labels)
│   └── metrics.json            # Evaluation metrics + training history
├── src/
│   ├── eda.py                  # Exploratory data analysis script
│   ├── preprocessing.py        # Text cleaning utilities
│   ├── train.py                # Model training script
│   └── app.py                  # Streamlit application
├── outputs/
│   ├── eda/                    # EDA plots (label distribution, correlations, etc.)
│   └── train_log.txt           # Training run log
└── README.md
```

## Approach

### 1. Data Exploration and Preparation
- Loaded and explored the Jigsaw comment toxicity dataset (159,571 labeled
  comments, 6 binary toxicity labels, ~10% positive rate — a highly
  imbalanced multi-label classification problem).
- Text preprocessing: lowercasing, URL/IP stripping, newline removal,
  punctuation normalization.
- Tokenization with a 20,000-word vocabulary and padded/truncated sequences
  (max length 150 tokens).

### 2. Model Development
- **Architecture**: Embedding (128-dim, trainable) → Bidirectional LSTM (64
  units) → GlobalMaxPooling1D → Dense(64, ReLU) → Dropout(0.3) →
  Dense(6, Sigmoid) for multi-label output.
- **Loss**: Binary cross-entropy (independent per-label sigmoid outputs).
- **Optimizer**: Adam (lr=1e-3).
- **Training**: 90/10 train/validation split, batch size 256, up to 6 epochs
  with early stopping on validation ROC-AUC.
- **Why Bi-LSTM over BERT**: given CPU-only training constraints, a
  Bidirectional LSTM offers a strong accuracy/latency/training-time
  trade-off for this dataset size. The architecture is modular, so it can be
  swapped for a transformer encoder (e.g. DistilBERT) if GPU resources are
  available — see `src/train.py::build_model`.

### 3. Streamlit Application
- **Real-time Prediction tab**: type/paste a comment, get per-label
  probability scores with an adjustable decision threshold.
- **Bulk CSV Prediction tab**: upload a CSV with a `comment_text` column,
  get per-label probabilities for every row, download results.
- **Model Insights tab**: ROC-AUC / F1 metrics, training curves, and EDA
  visualizations (label distribution, correlation heatmap, comment length).

## Setup & Usage

### 1. Install dependencies
```bash
pip install tensorflow-cpu streamlit pandas numpy scikit-learn matplotlib
```

### 2. Run EDA (optional, already generates outputs/eda/*.png)
```bash
cd src
python3 eda.py
```

### 3. Train the model
```bash
cd src
python3 train.py
```
This saves `toxicity_model.keras`, `tokenizer.pkl`, `config.json`, and
`metrics.json` to the `models/` directory.

### 4. Launch the Streamlit app
```bash
cd src
streamlit run app.py
```
Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Model Evaluation (Final Results)

Trained on 159,571 comments (90/10 train/validation split), early-stopped
after 4 epochs on validation ROC-AUC:

| Label | ROC-AUC |
|---|---|
| toxic | 0.978 |
| severe_toxic | 0.988 |
| obscene | 0.990 |
| threat | 0.962 |
| insult | 0.983 |
| identity_hate | 0.961 |
| **Mean AUC** | **0.977** |

Macro F1: 0.44 · Micro F1: 0.74 (at 0.5 decision threshold)

Because ~90% of comments are non-toxic, ROC-AUC (rather than raw accuracy)
is used as the primary metric to account for class imbalance. The `threat`
and `identity_hate` labels score lowest, consistent with them being the
rarest classes in training (478 and 1,405 positive examples respectively) —
a natural target for future improvement via class weighting or oversampling.
Full results, including epoch-by-epoch history, are in `models/metrics.json`.

## Technical Tags
Python, Deep Learning, Keras/TensorFlow, Bidirectional LSTM, NLP, Multi-label
Classification, Model Evaluation, Streamlit, Model Deployment

## Coding Standards
Code follows [PEP-8](https://www.python.org/dev/peps/pep-0008/) and is
organized into modular, reusable functions (`preprocessing.py`, `eda.py`,
`train.py`, `app.py`) so it can be maintained and extended independently.
