# 📄 Ticket Triage with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Project-green)]()
[![NLP](https://img.shields.io/badge/NLP-Text%20Processing-yellow)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)]()
[![TF-IDF](https://img.shields.io/badge/TF--IDF-Vectorization-purple)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)]()
[![Text Classification](https://img.shields.io/badge/Task-Text%20Classification-brightgreen)]()
[![Educational Project](https://img.shields.io/badge/Type-Educational-lightgrey)]()

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
|-------|-------------|
| id | Ticket identifier |
| oggetto | Short title |
| descrizione | Full description |
| categoria | Ticket category (Amministrazione / Tecnico / Commerciale) |
| priorità | Operational priority (Bassa / Media / Alta) |

**300 tickets** total, balanced across categories (100 per class).  
The text used by the model is obtained by combining **title + description**.

---

## 🧠 Machine Learning Approach

This project uses **Supervised Learning**.

### Text Processing

- Text is converted into numbers using **TF-IDF vectorization**
- Highlights important words while reducing common ones
- Uses unigrams and bigrams (`ngram_range=(1,2)`)

### Classification Model

The model used is **Logistic Regression**, chosen because:

✔ Simple and interpretable  
✔ Works well with small datasets  
✔ Fast to train  
✔ Suitable for text classification

Two independent models are trained:

- One for **Category** (Amministrazione / Tecnico / Commerciale)
- One for **Priority** (Bassa / Media / Alta)

---

## 📊 Evaluation Metrics

Model performance is evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score (macro)**
- **Confusion Matrix**

---

## 🖥️ Web Interface

A **Streamlit dashboard** with three sections:

### 1️⃣ Single Ticket Prediction
- Insert ticket title and description
- Get predicted category and priority
- View confidence scores
- See the **top 5 most influential words** that drove the classification

### 2️⃣ Batch Prediction (CSV)
- Upload a CSV file with multiple tickets (`oggetto` + `descrizione` columns)
- Run predictions on all tickets at once
- Download results as CSV with predicted category, priority and confidence scores

### 3️⃣ Model Evaluation
- View accuracy and F1-score macro on the test set
- Visualize per-class metrics (Precision, Recall, F1) as bar charts
- View **confusion matrices** for both category and priority
- Download the full test set predictions as CSV

---

## ⚙️ How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Wikilangelo/ticket-triage-ml.git
cd ticket-triage-ml
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ (Optional) Explore the notebooks

```bash
jupyter notebook
```

Open `Preprocessing.ipynb` to see the data preparation phase and `Training.ipynb` for model training and confusion matrix, with step-by-step explanations in Italian.

### 4️⃣ Train the models (optional, if .pkl files are missing)

```bash
python main.py
```

### 5️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📌 Project Structure

```
├── Preprocessing.ipynb       → Data cleaning and preparation
├── Training.ipynb            → Model training, evaluation and confusion matrix
├── app.py                    → Streamlit dashboard (3 sections)
├── main.py                   → Model training script
├── plot.py                   → Standalone evaluation plots
├── dataset_tickets_pw18.csv  → Synthetic dataset (300 tickets)
├── predizioni_test.csv       → Test set predictions
├── report_categoria.json     → Category metrics report
├── report_priorita.json      → Priority metrics report
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

This is a **prototype**, not a production system.

- Dataset is small (300 tickets) and synthetic
- No stopword removal — common Italian words may appear in top influential words
- Priority class **Alta** is underrepresented (46 tickets vs 162 Bassa), causing lower recall
- Language complexity is limited compared to real-world tickets

---

## 🚀 Future Improvements

- Use real-world datasets with more tickets
- Add Italian stopword removal
- Integrate advanced NLP models (e.g. BERT)
- Continuous model retraining pipeline
- Integration with real ticketing systems (Jira, Zendesk)

---

## 🎓 Academic Context

This project was developed as part of a university Project Work in **Machine Learning for Business Processes**, focusing on simplicity, clarity, and reproducibility.

---

## 🤝 Human + AI Collaboration

The system is designed to **support**, not replace, human operators.  
It provides initial classification while leaving final decisions to people.

---

**Author:** Michelangelo Bonvini
