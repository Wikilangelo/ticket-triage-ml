import streamlit as st
import pickle

model_cat = pickle.load(open("model_categoria.pkl", "rb"))
model_prio = pickle.load(open("model_priorita.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Triage Automatico Ticket")

testo = st.text_area("Inserisci oggetto e descrizione del ticket")


if st.button("Analizza"):
    if testo.strip() == "":
        st.error("Inserisci un testo prima di analizzare")
    else:
        testo_vec = vectorizer.transform([testo])
        categoria = model_cat.predict(testo_vec)[0]
        priorita = model_prio.predict(testo_vec)[0]

        st.success(f"Categoria: {categoria}")
        st.warning(f"Priorità: {priorita}")
