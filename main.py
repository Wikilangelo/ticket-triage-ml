import re
import json
import pickle
import unicodedata

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# ----------------------------
# Utility: pulizia del testo
# ----------------------------
def clean_text(s: str) -> str:
    """
    Pulizia testuale coerente con quanto descritto in tesi:
    - normalizzazione unicode (encoding/accents)
    - minuscole
    - rimozione caratteri speciali non utili
    - compressione spazi
    """
    if s is None:
        return ""
    s = str(s)

    # normalizzazione unicode (gestione encoding / caratteri)
    s = unicodedata.normalize("NFKC", s)

    # minuscole
    s = s.lower()

    # sostituisce newline/tab con spazio
    s = s.replace("\n", " ").replace("\t", " ")

    # rimuove caratteri speciali "rumorosi" mantenendo lettere/numeri/spazi e vocali accentate
    s = re.sub(r"[^\w\sàèéìòù]", " ", s, flags=re.UNICODE)

    # comprime spazi multipli
    s = re.sub(r"\s+", " ", s).strip()

    return s


def main():
    # 1) Caricamento dataset
    df = pd.read_csv("dataset_tickets_pw18.csv")

    # 2) Pulizia: valori mancanti + pulizia testuale
    df["oggetto"] = df["oggetto"].fillna("").apply(clean_text)
    df["descrizione"] = df["descrizione"].fillna("").apply(clean_text)

    # 3) Rimozione duplicati (coerente con tesi)
    #    duplicato = stesso oggetto+descrizione
    df = df.drop_duplicates(subset=["oggetto", "descrizione"]).reset_index(drop=True)

    # 4) Unione testo (oggetto + descrizione)
    df["testo"] = (df["oggetto"] + " " + df["descrizione"]).str.strip()

    # 5) Dataset "text/label" espliciti (coerente con tesi)
    df_cat = df[["testo", "categoria"]].rename(columns={"testo": "text", "categoria": "label"})
    df_prio = df[["testo", "priorita"]].rename(columns={"testo": "text", "priorita": "label"})

    # 6) Split UNICO 80/20: stesso ticket -> categoria e priorità coerenti
    X_train, X_test, y_train_cat, y_test_cat, y_train_prio, y_test_prio = train_test_split(
        df["testo"],
        df_cat["label"],
        df_prio["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["categoria"]  # aiuta a mantenere distribuzione per categoria
    )

    # 7) TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),     # migliora leggermente senza complicare troppo
        min_df=1,
        max_df=0.95
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 8) Modelli: Logistic Regression
    model_cat = LogisticRegression(max_iter=1000)
    model_prio = LogisticRegression(max_iter=1000)

    model_cat.fit(X_train_vec, y_train_cat)
    model_prio.fit(X_train_vec, y_train_prio)

    # 9) Predizioni
    pred_cat = model_cat.predict(X_test_vec)
    pred_prio = model_prio.predict(X_test_vec)

    # 10) Valutazione
    print("\n=== CATEGORIA ===")
    print("Accuracy:", accuracy_score(y_test_cat, pred_cat))
    print(classification_report(y_test_cat, pred_cat))

    print("\n=== PRIORITÀ ===")
    print("Accuracy:", accuracy_score(y_test_prio, pred_prio))
    print(classification_report(y_test_prio, pred_prio))

    # F1 macro per confronto rapido
    report_cat = classification_report(y_test_cat, pred_cat, output_dict=True)
    report_prio = classification_report(y_test_prio, pred_prio, output_dict=True)

    print("F1 macro (Categoria):", report_cat["macro avg"]["f1-score"])
    print("F1 macro (Priorità):", report_prio["macro avg"]["f1-score"])

    # 11) Salvataggio modelli e vectorizer per Streamlit
    pickle.dump(model_cat, open("model_categoria.pkl", "wb"))
    pickle.dump(model_prio, open("model_priorita.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    print("\nModelli salvati!")

    # 12) Output CSV batch con predizioni (utile per consegna)
    out = pd.DataFrame({
        "testo": X_test,
        "categoria_reale": y_test_cat,
        "categoria_predetta": pred_cat,
        "priorita_reale": y_test_prio,
        "priorita_predetta": pred_prio
    })
    out.to_csv("predizioni_test.csv", index=False, encoding="utf-8")
    print("Salvato predizioni_test.csv")

    # 13) Salvataggio metriche per visualizzazione grafica
    with open("report_categoria.json", "w", encoding="utf-8") as f:
        json.dump(report_cat, f, indent=4, ensure_ascii=False)

    with open("report_priorita.json", "w", encoding="utf-8") as f:
        json.dump(report_prio, f, indent=4, ensure_ascii=False)

    print("Salvati report_categoria.json e report_priorita.json")


if __name__ == "__main__":
    main()
