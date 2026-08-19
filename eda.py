import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def run_eda(train_path=r"C:\Users\shres\OneDrive\New folder\OneDrive\Desktop\ALL_PTOJECTS\5.comment_toxicity_project_1\src\train.csv", out_dir="outputs/eda"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(train_path)

    report = []
    report.append(f"Dataset shape: {df.shape}")
    report.append(f"Missing values:\n{df.isnull().sum().to_string()}")

    label_counts = df[LABELS].sum().sort_values(ascending=False)
    report.append(f"\nLabel counts:\n{label_counts.to_string()}")

    any_toxic = (df[LABELS].sum(axis=1) > 0).sum()
    clean = (df[LABELS].sum(axis=1) == 0).sum()
    report.append(f"\nAny toxic label: {any_toxic} ({any_toxic/len(df)*100:.2f}%)")
    report.append(f"Clean comments: {clean} ({clean/len(df)*100:.2f}%)")

    df["text_len"] = df["comment_text"].str.len()
    report.append(f"\nComment length stats:\n{df['text_len'].describe().to_string()}")

    # Plot 1: Label distribution
    plt.figure(figsize=(8, 5))
    label_counts.plot(kind="bar", color="firebrick")
    plt.title("Toxicity Label Counts (Train Set)")
    plt.ylabel("Number of Comments")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "label_distribution.png"), dpi=120)
    plt.close()

    # Plot 2: Clean vs toxic pie
    plt.figure(figsize=(5, 5))
    plt.pie([clean, any_toxic], labels=["Clean", "Toxic (any label)"],
            autopct="%1.1f%%", colors=["#4CAF50", "#E53935"])
    plt.title("Clean vs Toxic Comments")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "clean_vs_toxic.png"), dpi=120)
    plt.close()

    # Plot 3: Comment length histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df["text_len"].clip(upper=1000), bins=50, color="steelblue")
    plt.title("Comment Length Distribution (clipped at 1000 chars)")
    plt.xlabel("Character length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comment_length_hist.png"), dpi=120)
    plt.close()

    # Plot 4: Label co-occurrence heatmap
    corr = df[LABELS].corr()
    plt.figure(figsize=(6, 5))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    plt.yticks(range(len(LABELS)), LABELS)
    plt.colorbar()
    plt.title("Label Co-occurrence Correlation")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "label_correlation.png"), dpi=120)
    plt.close()

    with open(os.path.join(out_dir, "eda_report.txt"), "w") as f:
        f.write("\n".join(report))

    print("\n".join(report))
    print(f"\nEDA plots saved to {out_dir}/")

