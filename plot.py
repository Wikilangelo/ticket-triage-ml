import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

def plot_report(report_path, titolo):
    # 1) Carica il report dal file JSON
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    # 2) Prende solo le classi (esclude righe riassuntive)
    classi = [k for k in report.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
    # 3) Estrae precision, recall e f1-score per ogni classe
    precision = [report[c]["precision"] for c in classi]
    recall = [report[c]["recall"] for c in classi]
    f1_score = [report[c]["f1-score"] for c in classi]
    x = np.arange(len(classi))
    width = 0.25
    # 4) Grafico a barre affiancate
    plt.figure(figsize=(8, 5))
    plt.bar(x - width, precision, width, label="Precisione")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1_score, width, label="F1-Score")
    plt.ylabel("Valori")
    plt.title(titolo)
    plt.xticks(x, classi)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classi, titolo):
    """Genera e mostra la matrice di confusione."""
    cm = confusion_matrix(y_true, y_pred, labels=classi)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classi)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(titolo)
    plt.tight_layout()
    plt.show()


# --- Ricostruisce le predizioni sul test set per generare le confusion matrix ---
import re
import unicodedata
from sklearn.model_selection import train_test_split

def clean_text(s):
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"[^\w\sàèéìòù]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df = pd.read_csv("dataset_tickets_pw18.csv")
df["oggetto"] = df["oggetto"].fillna("").apply(clean_text)
df["descrizione"] = df["descrizione"].fillna("").apply(clean_text)
df = df.drop_duplicates(subset=["oggetto", "descrizione"]).reset_index(drop=True)
df["testo"] = (df["oggetto"] + " " + df["descrizione"]).str.strip()

_, X_test, _, y_test_cat, _, y_test_prio = train_test_split(
    df["testo"], df["categoria"], df["priorita"],
    test_size=0.2, random_state=42, stratify=df["categoria"]
)

model_cat  = pickle.load(open("model_categoria.pkl", "rb"))
model_prio = pickle.load(open("model_priorita.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

X_test_vec = vectorizer.transform(X_test)
pred_cat   = model_cat.predict(X_test_vec)
pred_prio  = model_prio.predict(X_test_vec)

# --- Grafici ---

# Grafico a barre per Priorità
plot_report("report_priorita.json", "Metriche per classe - Priorità")

# Grafico a barre per Categoria
plot_report("report_categoria.json", "Metriche per classe - Categoria")

# Confusion matrix per Categoria
plot_confusion_matrix(
    y_test_cat, pred_cat,
    classi=model_cat.classes_,
    titolo="Matrice di Confusione - Categoria"
)

# Confusion matrix per Priorità
plot_confusion_matrix(
    y_test_prio, pred_prio,
    classi=model_prio.classes_,
    titolo="Matrice di Confusione - Priorità"
)
