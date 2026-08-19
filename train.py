"""
Train a Bidirectional LSTM deep learning model for multi-label
comment toxicity classification.

Usage:
    python3 train.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, f1_score

from preprocessing import clean_text

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

VOCAB_SIZE = 20000
MAX_LEN = 150
EMBEDDING_DIM = 128

MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path=r"C:\Users\shres\OneDrive\New folder\OneDrive\Desktop\ALL_PTOJECTS\5.comment_toxicity_project_1\src\train.csv"):
    df = pd.read_csv(path)
    df["clean_text"] = df["comment_text"].apply(clean_text)
    return df


def build_model(vocab_size, embedding_dim, max_len, n_labels):
    model = keras.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
        layers.Bidirectional(
            layers.LSTM(64, return_sequences=True)
        ),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_labels, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc", multi_label=True)
        ],
    )

    return model


def main():
    print("Loading data...")
    df = load_data()

    X_text = np.array(df["clean_text"].tolist(), dtype=object)
    y = df[LABELS].values.astype("float32")

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_text,
        y,
        test_size=0.1,
        random_state=42
    )

    print("Fitting tokenizer...")

    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(X_train_text)

    X_train_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_train_text),
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    X_val_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_val_text),
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    print("Building model...")

    model = build_model(
        VOCAB_SIZE,
        EMBEDDING_DIM,
        MAX_LEN,
        len(LABELS)
    )

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=2,
            restore_best_weights=True
        )
    ]

    print("Training...")

    history = model.fit(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=6,
        batch_size=256,
        callbacks=callbacks,
        verbose=2,
    )

    print("Evaluating...")

    y_pred_proba = model.predict(
        X_val_seq,
        batch_size=512
    )

    y_pred = (y_pred_proba > 0.5).astype(int)

    auc_scores = {}

    for i, label in enumerate(LABELS):
        try:
            auc_scores[label] = roc_auc_score(
                y_val[:, i],
                y_pred_proba[:, i]
            )
        except ValueError:
            auc_scores[label] = None

    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    micro_f1 = f1_score(
        y_val,
        y_pred,
        average="micro",
        zero_division=0
    )

    metrics = {
        "auc_per_label": auc_scores,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "mean_auc": float(
            np.mean(
                [v for v in auc_scores.values() if v is not None]
            )
        ),
        "history": {
            k: [float(x) for x in v]
            for k, v in history.history.items()
        },
    }

    print(json.dumps(metrics, indent=2))

    # Save model and artifacts
    model.save(os.path.join(MODEL_DIR, "toxicity_model.keras"))

    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump(
            {
                "vocab_size": VOCAB_SIZE,
                "max_len": MAX_LEN,
                "embedding_dim": EMBEDDING_DIM,
                "labels": LABELS,
            },
            f,
            indent=2,
        )

    print("Saved model, tokenizer, metrics, config to", MODEL_DIR)


if __name__ == "__main__":
    main()

