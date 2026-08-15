"""
train.py
--------
End-to-end pipeline:
  1. Load paired image+text dataset (dataset/data.csv)
  2. Preprocess: image resize/normalise, text clean + TF-IDF vectorise
  3. Stratified train / val / test split (70 / 15 / 15)
  4. Train three models: Image-only, Text-only, Fusion (image+text)
  5. Evaluate all three on the held-out test set with Accuracy, Precision,
     Recall, F1 (macro) and confusion matrices
  6. Save metrics, plots, and a qualitative prediction sample to
     project/outputs and project/figures
"""
import os
import re
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ImageOnlyClassifier, TextOnlyClassifier, FusionClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA_CSV = os.path.join(BASE, "dataset", "data.csv")
DATA_ROOT = os.path.join(BASE, "dataset")
OUT_DIR = os.path.join(BASE, "outputs")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 18
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------- Text preprocessing --------------------------- #
def clean_text(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ------------------------------- Dataset ----------------------------------- #
class ProductDataset(Dataset):
    def __init__(self, df, tfidf_matrix, label_ids):
        self.df = df.reset_index(drop=True)
        self.tfidf = tfidf_matrix
        self.labels = label_ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(DATA_ROOT, row["image_path"])
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5                       # normalise to [-1, 1]
        arr = np.transpose(arr, (2, 0, 1))             # HWC -> CHW
        img_tensor = torch.tensor(arr, dtype=torch.float32)

        text_vec = torch.tensor(self.tfidf[idx].toarray()[0], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, text_vec, label


# ------------------------------ Train / Eval -------------------------------- #
def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for img, txt, label in loader:
            img, txt, label = img.to(DEVICE), txt.to(DEVICE), label.to(DEVICE)
            if train:
                optimizer.zero_grad()
            logits = model(img, txt)
            loss = criterion(logits, label)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * label.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


def train_model(model, train_loader, val_loader, name, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, optimizer, criterion, train=False)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{name}] epoch {epoch:2d}/{epochs}  "
              f"train_loss={tr_loss:.3f} train_acc={tr_acc:.3f}  "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

    model.load_state_dict(best_state)
    return model, history


def evaluate(model, loader, label_encoder, name):
    criterion = nn.CrossEntropyLoss()
    _, acc, preds, labels = run_epoch(model, loader, None, criterion, train=False)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds)
    report = classification_report(
        labels, preds, target_names=label_encoder.classes_, zero_division=0)
    metrics = {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }
    print(f"\n=== {name} — Test Set ===")
    print(json.dumps(metrics, indent=2))
    print(report)
    return metrics, cm, report


def plot_confusion(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, h in histories.items():
        axes[0].plot(h["val_acc"], label=name)
        axes[1].plot(h["val_loss"], label=name)
    axes[0].set_title("Validation Accuracy per Epoch"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].set_title("Validation Loss per Epoch"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison_bar(metrics_by_model, path):
    labels = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    models = list(metrics_by_model.keys())
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, m in enumerate(models):
        vals = [metrics_by_model[m][l] for l in labels]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Unimodal Baselines vs. Image+Text Fusion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_CSV)
    df["clean_text"] = df["text"].apply(clean_text)

    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])

    # Stratified 70/15/15 split
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label_id"], random_state=SEED)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=SEED)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # TF-IDF fit ONLY on train text (avoid leakage)
    vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    tfidf_train = vectorizer.fit_transform(train_df["clean_text"])
    tfidf_val = vectorizer.transform(val_df["clean_text"])
    tfidf_test = vectorizer.transform(test_df["clean_text"])

    train_ds = ProductDataset(train_df, tfidf_train, train_df["label_id"].to_numpy())
    val_ds = ProductDataset(val_df, tfidf_val, val_df["label_id"].to_numpy())
    test_ds = ProductDataset(test_df, tfidf_test, test_df["label_id"].to_numpy())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(le.classes_)
    tfidf_dim = tfidf_train.shape[1]

    results = {}
    histories = {}
    confusions = {}
    reports = {}

    # ---- Image-only baseline ----
    img_model = ImageOnlyClassifier(num_classes)
    img_model, hist = train_model(img_model, train_loader, val_loader, "Image-only")
    m, cm, rep = evaluate(img_model, test_loader, le, "Image-only")
    results["Image-only"] = m; histories["Image-only"] = hist
    confusions["Image-only"] = cm; reports["Image-only"] = rep

    # ---- Text-only baseline ----
    txt_model = TextOnlyClassifier(num_classes, tfidf_dim)
    txt_model, hist = train_model(txt_model, train_loader, val_loader, "Text-only")
    m, cm, rep = evaluate(txt_model, test_loader, le, "Text-only")
    results["Text-only"] = m; histories["Text-only"] = hist
    confusions["Text-only"] = cm; reports["Text-only"] = rep

    # ---- Fusion model ----
    fus_model = FusionClassifier(num_classes, tfidf_dim)
    fus_model, hist = train_model(fus_model, train_loader, val_loader, "Image+Text Fusion")
    m, cm, rep = evaluate(fus_model, test_loader, le, "Image+Text Fusion")
    results["Image+Text Fusion"] = m; histories["Image+Text Fusion"] = hist
    confusions["Image+Text Fusion"] = cm; reports["Image+Text Fusion"] = rep

    # ---- Save artefacts ----
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT_DIR, "classification_reports.txt"), "w") as f:
        for name, rep in reports.items():
            f.write(f"=== {name} ===\n{rep}\n\n")

    for name, cm in confusions.items():
        safe = name.replace(" ", "_").replace("+", "")
        plot_confusion(cm, le.classes_, f"Confusion Matrix — {name}",
                        os.path.join(FIG_DIR, f"confusion_{safe}.png"))

    plot_history(histories, os.path.join(FIG_DIR, "training_curves.png"))
    plot_comparison_bar(results, os.path.join(FIG_DIR, "model_comparison.png"))

    # qualitative predictions sample from the fusion model
    fus_model.eval()
    sample = test_df.sample(n=min(10, len(test_df)), random_state=SEED).reset_index(drop=True)
    sample_tfidf = vectorizer.transform(sample["clean_text"])
    rows = []
    with torch.no_grad():
        for i in range(len(sample)):
            img = Image.open(os.path.join(DATA_ROOT, sample.loc[i, "image_path"])).convert("RGB")
            arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
            arr = np.transpose(arr, (2, 0, 1))
            img_t = torch.tensor(arr).unsqueeze(0).to(DEVICE)
            txt_t = torch.tensor(sample_tfidf[i].toarray(), dtype=torch.float32).to(DEVICE)
            logits = fus_model(img_t, txt_t)
            pred = le.classes_[logits.argmax(dim=1).item()]
            rows.append({
                "text": sample.loc[i, "text"],
                "true_label": sample.loc[i, "label"],
                "predicted_label": pred,
                "correct": pred == sample.loc[i, "label"],
            })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "sample_predictions.csv"), index=False)

    with open(os.path.join(OUT_DIR, "dataset_split_summary.json"), "w") as f:
        json.dump({
            "total_samples": len(df),
            "train": len(train_df), "val": len(val_df), "test": len(test_df),
            "classes": list(le.classes_),
            "tfidf_vocab_size": tfidf_dim,
        }, f, indent=2)

    print("\nAll metrics, plots and sample predictions saved to /outputs and /figures")


if __name__ == "__main__":
    main()
