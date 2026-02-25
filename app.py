import re
import unicodedata
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

# ============================================================
# Caricamento modelli e vectorizer
# ============================================================
model_cat  = pickle.load(open("model_categoria.pkl", "rb"))
model_prio = pickle.load(open("model_priorita.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

MIN_CHARS = 30
MIN_WORDS = 5


def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"[^\w\sàèéìòù]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_top_words(testo_vec, model, n=5):
    feature_names = np.array(vectorizer.get_feature_names_out())
    classe_predetta_idx = model.predict(testo_vec)[0]
    classe_idx = list(model.classes_).index(classe_predetta_idx)
    coef = model.coef_[classe_idx]
    tfidf_scores = testo_vec.toarray()[0]
    contributo = coef * tfidf_scores
    top_idx = np.argsort(contributo)[::-1][:n]
    return [(feature_names[i], contributo[i]) for i in top_idx if contributo[i] > 0]


# ============================================================
# TITOLO
# ============================================================
st.title("Sistema di Triage Automatico dei Ticket")

# ============================================================
# SEZIONE 1 — PREDIZIONE SINGOLO TICKET
# ============================================================
st.header("Predizione singolo ticket")
st.write("Inserisci oggetto e descrizione del ticket per ottenere categoria e priorità suggerite.")

oggetto    = st.text_input("Oggetto del ticket")
descrizione = st.text_area("Descrizione del ticket")

if st.button("Analizza ticket"):
    testo = (oggetto + " " + descrizione).strip()

    if testo == "":
        st.error("Inserisci oggetto e descrizione prima di analizzare.")
    elif len(testo) < MIN_CHARS or len(testo.split()) < MIN_WORDS:
        st.warning("Il testo è troppo corto. Inserisci più dettagli per ottenere una previsione affidabile.")
    else:
        testo_vec = vectorizer.transform([testo])

        proba_cat  = model_cat.predict_proba(testo_vec)[0]
        idx_cat    = proba_cat.argmax()
        categoria  = model_cat.classes_[idx_cat]
        conf_cat   = proba_cat[idx_cat]

        proba_prio = model_prio.predict_proba(testo_vec)[0]
        idx_prio   = proba_prio.argmax()
        priorita   = model_prio.classes_[idx_prio]
        conf_prio  = proba_prio[idx_prio]

        st.success(f"Categoria prevista: **{categoria}**")
        st.info(f"Confidenza categoria: {conf_cat:.2%}")
        st.warning(f"Priorità prevista: **{priorita}**")
        st.info(f"Confidenza priorità: {conf_prio:.2%}")

        st.divider()
        st.subheader("Parole più influenti per la categoria")
        top = get_top_words(testo_vec, model_cat, n=5)
        if top:
            for parola, peso in top:
                st.write(f"• **{parola}** (peso: {peso:.4f})")
        else:
            st.write("Nessuna parola sufficientemente influente trovata.")

# ============================================================
# SEZIONE 2 — PREDIZIONE BATCH (CSV)
# ============================================================
st.divider()
st.header("Predizione batch (CSV)")
st.write("Carica un file CSV con i ticket per ottenere le predizioni su tutti in una volta.")

with st.expander("Formato del file CSV richiesto"):
    st.markdown("""
Il file CSV deve contenere almeno queste due colonne:
- **oggetto** → breve titolo del ticket
- **descrizione** → testo completo del ticket

Esempio:
```
oggetto,descrizione
Problema login,Non riesco ad accedere al portale aziendale
Richiesta ferie,Vorrei richiedere ferie per la settimana prossima
```
""")

csv_file = st.file_uploader("Carica CSV", type=["csv"])

if csv_file is not None:
    try:
        df_batch = pd.read_csv(csv_file)

        if "oggetto" not in df_batch.columns or "descrizione" not in df_batch.columns:
            st.error("Il file CSV deve contenere le colonne 'oggetto' e 'descrizione'.")
        else:
            st.success(f"File caricato: {len(df_batch)} ticket trovati.")
            st.subheader("Anteprima")
            st.dataframe(df_batch.head(5))

            if st.button("Avvia predizione batch"):
                with st.spinner("Elaborazione in corso..."):
                    df_batch["oggetto"]     = df_batch["oggetto"].fillna("").apply(clean_text)
                    df_batch["descrizione"] = df_batch["descrizione"].fillna("").apply(clean_text)
                    df_batch["testo"]       = (df_batch["oggetto"] + " " + df_batch["descrizione"]).str.strip()

                    X_batch = vectorizer.transform(df_batch["testo"])

                    df_batch["categoria_prevista"]  = model_cat.predict(X_batch)
                    df_batch["priorita_prevista"]   = model_prio.predict(X_batch)
                    df_batch["confidenza_categoria"] = model_cat.predict_proba(X_batch).max(axis=1).round(4)
                    df_batch["confidenza_priorita"]  = model_prio.predict_proba(X_batch).max(axis=1).round(4)
                    df_risultati = df_batch.drop(columns=["testo"])

                st.success("Predizione completata!")
                st.subheader("Risultati")
                st.dataframe(df_risultati)

                csv_output = df_risultati.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Scarica CSV con predizioni",
                    data=csv_output,
                    file_name="predizioni_batch.csv",
                    mime="text/csv"
                )
    except Exception as e:
        st.error(f"Errore nella lettura del file: {e}")

# ============================================================
# SEZIONE 3 — VALUTAZIONE DEL MODELLO
# ============================================================
st.divider()
st.header("Valutazione del modello")
st.write("Metriche di valutazione calcolate sul test set (20% del dataset originale).")

if st.button("Mostra valutazione"):
    with st.spinner("Caricamento valutazione..."):

        # Ricostruisce il test set
        df_eval = pd.read_csv("dataset_tickets_pw18.csv")
        df_eval["oggetto"]     = df_eval["oggetto"].fillna("").apply(clean_text)
        df_eval["descrizione"] = df_eval["descrizione"].fillna("").apply(clean_text)
        df_eval = df_eval.drop_duplicates(subset=["oggetto", "descrizione"]).reset_index(drop=True)
        df_eval["testo"] = (df_eval["oggetto"] + " " + df_eval["descrizione"]).str.strip()

        _, X_test, _, y_test_cat, _, y_test_prio = train_test_split(
            df_eval["testo"], df_eval["categoria"], df_eval["priorita"],
            test_size=0.2, random_state=42, stratify=df_eval["categoria"]
        )

        X_test_vec = vectorizer.transform(X_test)
        pred_cat   = model_cat.predict(X_test_vec)
        pred_prio  = model_prio.predict(X_test_vec)

    # --- CATEGORIA ---
    st.subheader("Categoria")

    rep_cat = classification_report(y_test_cat, pred_cat, output_dict=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{rep_cat['accuracy']:.2%}")
    with col2:
        st.metric("F1-score macro", f"{rep_cat['macro avg']['f1-score']:.2%}")

    # Grafico a barre metriche categoria
    classi_cat = [k for k in rep_cat if k not in ["accuracy", "macro avg", "weighted avg"]]
    x = np.arange(len(classi_cat))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, [rep_cat[c]["precision"] for c in classi_cat], width, label="Precisione")
    ax.bar(x,         [rep_cat[c]["recall"]    for c in classi_cat], width, label="Recall")
    ax.bar(x + width, [rep_cat[c]["f1-score"]  for c in classi_cat], width, label="F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels(classi_cat)
    ax.set_ylim(0, 1)
    ax.set_title("Metriche per classe - Categoria")
    ax.set_ylabel("Valori")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

    # Confusion matrix categoria
    cm_cat = confusion_matrix(y_test_cat, pred_cat, labels=model_cat.classes_)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm_cat, display_labels=model_cat.classes_).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Matrice di Confusione - Categoria")
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # --- PRIORITÀ ---
    st.subheader("Priorità")

    rep_prio = classification_report(y_test_prio, pred_prio, output_dict=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{rep_prio['accuracy']:.2%}")
    with col2:
        st.metric("F1-score macro", f"{rep_prio['macro avg']['f1-score']:.2%}")

    # Grafico a barre metriche priorità
    classi_prio = [k for k in rep_prio if k not in ["accuracy", "macro avg", "weighted avg"]]
    x = np.arange(len(classi_prio))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, [rep_prio[c]["precision"] for c in classi_prio], width, label="Precisione")
    ax.bar(x,         [rep_prio[c]["recall"]    for c in classi_prio], width, label="Recall")
    ax.bar(x + width, [rep_prio[c]["f1-score"]  for c in classi_prio], width, label="F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels(classi_prio)
    ax.set_ylim(0, 1)
    ax.set_title("Metriche per classe - Priorità")
    ax.set_ylabel("Valori")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

    # Confusion matrix priorità
    cm_prio = confusion_matrix(y_test_prio, pred_prio, labels=model_prio.classes_)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm_prio, display_labels=model_prio.classes_).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Matrice di Confusione - Priorità")
    plt.tight_layout()
    st.pyplot(fig)

    # Download risultati valutazione
    st.divider()
    st.subheader("Download")
    df_download = pd.DataFrame({
        "testo": X_test.values,
        "categoria_reale": y_test_cat.values,
        "categoria_prevista": pred_cat,
        "priorita_reale": y_test_prio.values,
        "priorita_prevista": pred_prio,
    })
    csv_val = df_download.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Scarica CSV con predizioni del test set",
        data=csv_val,
        file_name="predizioni_test.csv",
        mime="text/csv"
    )
