import pandas as pd
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# 1) Caricamento dataset
df = pd.read_csv("dataset_tickets_pw18.csv")

# 2) Pulizia minima (evita errori se ci sono celle vuote)
df["oggetto"] = df["oggetto"].fillna("")
df["descrizione"] = df["descrizione"].fillna("")

# 3) Unione testo (oggetto + descrizione)
df["testo"] = (df["oggetto"] + " " + df["descrizione"]).str.strip()

# 4) Split UNICO 80/20: stesso ticket -> categoria e priorità coerenti
X_train, X_test, y_train_cat, y_test_cat, y_train_prio, y_test_prio = train_test_split(
    df["testo"],
    df["categoria"],
    df["priorita"],
    test_size=0.2,
    random_state=42
)

# 5) TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6) Modelli (semplici)
model_cat = LogisticRegression(max_iter=1000)
model_prio = LogisticRegression(max_iter=1000)

model_cat.fit(X_train_vec, y_train_cat)
model_prio.fit(X_train_vec, y_train_prio)

# 7) Predizioni
pred_cat = model_cat.predict(X_test_vec)
pred_prio = model_prio.predict(X_test_vec)

# 8) Valutazione (accuracy + report con F1)
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

# 9) Salvataggio modelli e vectorizer necessari per Streamlit
pickle.dump(model_cat, open("model_categoria.pkl", "wb"))
pickle.dump(model_prio, open("model_priorita.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModelli salvati!")

# 10) Output CSV batch con predizioni (utile per consegna)
out = pd.DataFrame({
    "testo": X_test,
    "categoria_reale": y_test_cat,
    "categoria_predetta": pred_cat,
    "priorita_reale": y_test_prio,
    "priorita_predetta": pred_prio
})
out.to_csv("predizioni_test.csv", index=False)
print("Salvato predizioni_test.csv")

# 11) Salvataggio metriche per visualizzazione grafica
with open("report_categoria.json", "w", encoding="utf-8") as f:
    json.dump(report_cat, f, indent=4)

with open("report_priorita.json", "w", encoding="utf-8") as f:
    json.dump(report_prio, f, indent=4)

print("Salvati report_categoria.json e report_priorita.json")