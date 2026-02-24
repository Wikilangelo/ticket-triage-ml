import streamlit as st
import pickle

# Caricamento modelli e vectorizer
model_cat = pickle.load(open("model_categoria.pkl", "rb"))
model_prio = pickle.load(open("model_priorita.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Titolo e descrizione
st.title("Sistema di Triage Automatico dei Ticket")
st.write("Inserisci oggetto e descrizione del ticket per ottenere categoria e priorità suggerite.")

# Input separati (coerente con la tesi)
oggetto = st.text_input("Oggetto del ticket")
descrizione = st.text_area("Descrizione del ticket")

# Parametri minimi di validazione
MIN_CHARS = 30
MIN_WORDS = 5

if st.button("Analizza ticket"):

    testo = (oggetto + " " + descrizione).strip()

    # Validazioni
    if testo == "":
        st.error("Inserisci oggetto e descrizione prima di analizzare.")
    
    elif len(testo) < MIN_CHARS or len(testo.split()) < MIN_WORDS:
        st.warning(
            "Il testo è troppo corto. Inserisci più dettagli per ottenere una previsione affidabile."
        )

    else:
        # Vectorizzazione
        testo_vec = vectorizer.transform([testo])

        # Predizione categoria con probabilità
        proba_cat = model_cat.predict_proba(testo_vec)[0]
        classi_cat = model_cat.classes_
        idx_cat = proba_cat.argmax()

        categoria = classi_cat[idx_cat]
        conf_cat = proba_cat[idx_cat]

        # Predizione priorità con probabilità
        proba_prio = model_prio.predict_proba(testo_vec)[0]
        classi_prio = model_prio.classes_
        idx_prio = proba_prio.argmax()

        priorita = classi_prio[idx_prio]
        conf_prio = proba_prio[idx_prio]

        # Output
        st.success(f"Categoria prevista: {categoria}")
        st.info(f"Confidenza categoria: {conf_cat:.2%}")

        st.warning(f"Priorità prevista: {priorita}")
        st.info(f"Confidenza priorità: {conf_prio:.2%}")
