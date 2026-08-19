import os
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing import clean_text

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.set_page_config(
    page_title="Comment Toxicity Detector",
    layout="wide",
)

LABEL_INFO = {
    "toxic": "Generally rude, disrespectful, or unreasonable",
    "severe_toxic": "Extremely hateful or aggressive",
    "obscene": "Contains obscene / vulgar language",
    "threat": "Contains a threat of violence or harm",
    "insult": "Insulting toward a person or group",
    "identity_hate": "Hateful toward a protected identity/group",
}

@st.cache(allow_output_mutation=True)
#@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "toxicity_model.keras"))
    metrics = None
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return model, tokenizer, config, metrics


def predict(texts, model, tokenizer, config):
    cleaned = [clean_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seqs, maxlen=config["max_len"], padding="post", truncating="post")
    preds = model.predict(padded, verbose=0)
    return preds


def main():
    st.title("🛡️ Comment Toxicity Detector")
    st.caption("Deep Learning (Bidirectional LSTM) model for multi-label toxic comment classification")

    try:
        model, tokenizer, config, metrics = load_artifacts()
    except Exception as e:
        st.error(
            "Model artifacts not found. Please run `python3 src/train.py` first to "
            "train and save the model.\n\nDetails: " + str(e)
        )
        st.stop()

    labels = config["labels"]

    tab1, tab2, tab3 = st.tabs(["🔍 Real-time Prediction", "📁 Bulk CSV Prediction", "📊 Model Insights"])

    # Real-time single comment prediction 
    with tab1:
        st.subheader("Analyze a single comment")
        default_text = "You are an amazing contributor, thanks for your help!"
        text_input = st.text_area("Enter a comment to analyze:", value=default_text, height=120)
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)

        if st.button("Analyze Comment"):
            if not text_input.strip():
                st.warning("Please enter some text.")
            else:
                proba = predict([text_input], model, tokenizer, config)[0]
                is_toxic_any = bool((proba >= threshold).any())

                if is_toxic_any:
                    st.error("⚠️ This comment was flagged as potentially TOXIC")
                else:
                    st.success("✅ This comment appears non-toxic")

                st.write("### Prediction breakdown")
                cols = st.columns(len(labels))
                for i, label in enumerate(labels):
                    with cols[i]:
                        st.metric(label, f"{proba[i]*100:.1f}%")
                        st.progress(float(proba[i]))
                        st.caption(LABEL_INFO.get(label, ""))

        st.markdown("---")
        st.write("#### Try a sample comment")
        samples = [
            "Thanks so much for fixing this, really appreciate the effort!",
            "You are an idiot and everyone hates your stupid edits.",
            "I will find you and make you regret this.",
        ]
        sample_choice = st.selectbox("Sample comments", ["-- select --"] + samples)
        if sample_choice != "-- select --":
            proba = predict([sample_choice], model, tokenizer, config)[0]
            df_sample = pd.DataFrame({"label": labels, "probability": proba})
            st.bar_chart(df_sample.set_index("label"))

#Bulk CSV prediction 
    with tab2:
        st.subheader("Upload a CSV file for bulk predictions")
        st.write("The CSV must contain a column named **comment_text**.")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if "comment_text" not in df.columns:
                st.error("CSV must contain a 'comment_text' column.")
            else:
                with st.spinner(f"Scoring {len(df)} comments..."):
                    preds = predict(df["comment_text"].astype(str).tolist(), model, tokenizer, config)
                    pred_df = pd.DataFrame(preds, columns=[f"{l}_prob" for l in labels])
                    result_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
                    result_df["any_toxic"] = (pred_df.values >= 0.5).any(axis=1)

                st.success(f"Scored {len(df)} comments.")
                st.write(f"**{result_df['any_toxic'].sum()}** flagged as toxic "
                         f"({result_df['any_toxic'].mean()*1000:.1f}%)")
                st.dataframe(result_df.head(200))

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions as CSV",
                    data=csv_bytes,
                    file_name="toxicity_predictions.csv",
                    mime="text/csv",
                )

    #Model insights / dashboard 
    with tab3:
        st.subheader("Model performance & dataset insights")
        if metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean ROC-AUC", f"{metrics.get('mean_auc', 0):.3f}")
            c2.metric("Macro F1", f"{metrics.get('macro_f1', 0):.3f}")
            c3.metric("Micro F1", f"{metrics.get('micro_f1', 0):.3f}")

            st.write("#### Per-label ROC-AUC")
            auc_df = pd.DataFrame(
                list(metrics["auc_per_label"].items()), columns=["label", "roc_auc"]
            )
            st.bar_chart(auc_df.set_index("label"))

            if "history" in metrics:
                st.write("#### Training history")
                hist_df = pd.DataFrame(metrics["history"])
                st.line_chart(hist_df[[c for c in hist_df.columns if "loss" in c]])
                st.line_chart(hist_df[[c for c in hist_df.columns if "auc" in c]])
        else:
            st.info("No metrics.json found yet. Train the model to populate this dashboard.")

        eda_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "eda")
        if os.path.isdir(eda_dir):
            st.write("#### Dataset exploration")
            imgs = ["label_distribution.png", "clean_vs_toxic.png",
                    "comment_length_hist.png", "label_correlation.png"]
            cols = st.columns(2)
            for i, img in enumerate(imgs):
                path = os.path.join(eda_dir, img)
                if os.path.exists(path):
                    with cols[i % 2]:
                        st.image(path, use_column_width=True)


if __name__ == "__main__":
    main()
