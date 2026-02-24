# 📄 Ticket Triage with Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Project-green)
![NLP](https://img.shields.io/badge/NLP-Text%20Processing-yellow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![TF-IDF](https://img.shields.io/badge/TF--IDF-Vectorization-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![Text Classification](https://img.shields.io/badge/Task-Text%20Classification-brightgreen)
![Educational Project](https://img.shields.io/badge/Type-Educational-lightgrey)

---

## 📌 Project Overview

This project is a **Machine Learning prototype** developed for an academic Project Work.  
The goal is to automatically perform **ticket triage** by:

- Classifying tickets into the correct **category**
- Suggesting an **operational priority**

The system simulates a real business environment where companies receive many daily support requests.

---

## 🎯 Objectives

The prototype demonstrates how **basic ML techniques** can:

✔ Reduce manual ticket sorting  
✔ Improve response times  
✔ Standardize classification  
✔ Support human operators in repetitive tasks

---

## 📂 Dataset

The dataset is **synthetic**, created specifically for the project, as required by the assignment guidelines.

Each ticket includes:

| Field | Description |
|---|---|
| id | Ticket identifier |
| oggetto | Short title |
| descrizione | Full description |
| categoria | Ticket category (Admin / Technical / Commercial) |
| priorità | Operational priority (Low / Medium / High) |

The text used by the model is obtained by combining **title + description**.

---

## 🧠 Machine Learning Approach

This project uses **Supervised Learning**.

### Text Processing

- Text is converted into numbers using **TF-IDF vectorization**
- This highlights important words while reducing common ones

### Classification Model

The model used is **Logistic Regression**, chosen because:

✔ Simple and interpretable  
✔ Works well with small datasets  
✔ Fast to train  
✔ Suitable for text classification

Two models are trained:

- One for **Category**
- One for **Priority**

---

## 📊 Evaluation Metrics

Model performance is evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

These metrics ensure the model is reasonably reliable for a prototype.

---

## 🖥️ Web Interface

A simple **Streamlit dashboard** allows users to:

1. Insert ticket title and description
2. Get predicted category
3. Get predicted priority
4. View model confidence

---

## ⚙️ How to Run the Project

### 1️⃣ Clone the repository

```
git clone https://github.com/Wikilangelo/ticket-triage-ml.git
cd ticket-triage-ml
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ (Opzionale) Esplora i notebook

```
jupyter notebook
```

Apri `Preprocessing.ipynb` per vedere la fase di preparazione dei dati e `Training.ipynb` per il training del modello, con spiegazioni passo passo in italiano.

### 4️⃣ Train the models (optional, if needed)

```
python main.py
```

### 5️⃣ Run the Streamlit app

```
streamlit run app.py
```

---

## 📌 Project Structure

```
├── Preprocessing.ipynb       → Pulizia e preparazione del dataset
├── Training.ipynb            → Addestramento e valutazione del modello
├── app.py                    → Streamlit interface
├── main.py                   → Model training script
├── plot.py                   → Evaluation plots
├── dataset_tickets_pw18.csv
├── report_categoria.json
├── report_priorita.json
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

This is a **prototype**, not a production system.

- Dataset is small and synthetic
- Language complexity is limited
- High priority tickets should always be reviewed by humans

---

## 🚀 Future Improvements

- Use real-world datasets
- Integrate advanced NLP models
- Continuous model retraining
- Integration with real ticketing systems

---

## 🎓 Academic Context

This project was developed as part of a university Project Work in **Machine Learning for Business Processes**, focusing on simplicity, clarity, and reproducibility.

---

## 🤝 Human + AI Collaboration

The system is designed to **support**, not replace, human operators.  
It provides initial classification while leaving final decisions to people.

---

**Author:** Michelangelo Bonvini
