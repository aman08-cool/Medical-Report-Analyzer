# AI Medical Report Analyzer

A web-based application that helps users **understand medical reports** by extracting key information and generating a **simple, patient-friendly summary** using modern NLP techniques.

> ⚠️ **Important:** This tool is strictly for informational purposes.
> It does **not** provide medical advice, diagnosis, or treatment recommendations.

---

## 🚀 Live Demo

🔗 **[Add your Streamlit Cloud URL here]**

---

## 🧠 What this project does

Given a raw medical report (free-text), the system:

* Extracts **key entities** (e.g., dates, drugs, procedures, organizations)
* Generates a **readable summary** using a transformer-based model
* Clearly communicates **limitations and safety boundaries**
* Presents results through a simple, accessible web interface

The focus is on **document understanding**, not medical decision-making.

---

## 🏗️ Architecture (High Level)

* **Frontend / UI**

  * Streamlit (Python)
* **NLP & ML**

  * spaCy (entity extraction)
  * Transformer-based summarization (BART)
* **Model Handling**

  * Cached models for performance
  * Dynamic input-length handling
* **Ethics & Safety**

  * Explicit non-diagnostic scope
  * Prominent disclaimers

---

## 🛠️ Tech Stack

* Python
* Streamlit
* spaCy
* Hugging Face Transformers
* PyTorch

---

## 📦 Project Structure

```
ai-medical-report-analyzer/
│
├── app.py              # Streamlit application
├── requirements.txt    # Project dependencies
├── README.md
├── notebooks/          # Experiments & exploration
├── data/               # Dataset (optional)
└── .gitignore
```

---

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ai-medical-report-analyzer.git
cd ai-medical-report-analyzer
```

### 2. Create & activate virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## ⚠️ Disclaimer

This application is intended to **assist with understanding medical text only**.
It does **not** replace professional medical consultation, diagnosis, or treatment.

Always consult a qualified healthcare professional for medical concerns.

---

## 📌 Notes on Responsible AI

* No medical inference or diagnosis is performed
* Output is constrained and explainable
* Long inputs are safely handled to avoid hallucinations
* The system prioritizes clarity over speculation

---

## 📄 License

This project is released for **educational and community use**.

