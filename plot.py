import json
import numpy as np
import matplotlib.pyplot as plt


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

    plt.show()


# Grafico per Priorità
plot_report("report_priorita.json", "Metriche per classe - Priorità")

# Grafico per Categoria
plot_report("report_categoria.json", "Metriche per classe - Categoria")
