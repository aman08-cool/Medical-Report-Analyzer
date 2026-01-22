import streamlit as st
import spacy
from transformers import pipeline
from collections import defaultdict

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="AI Medical Report Analyzer",
    layout="centered"
)

# --------------------------------------------------
# Load models (cached for performance)
# --------------------------------------------------
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        framework="pt"
    )
    return nlp, summarizer

nlp, summarizer = load_models()

# --------------------------------------------------
# Constants
# --------------------------------------------------
DISCLAIMER = (
    "This summary is for informational purposes only and does not "
    "constitute medical advice or diagnosis."
)

TARGET_LABELS = {"DISEASE", "DRUG", "DATE", "PROCEDURE", "ORG"}

# --------------------------------------------------
# NLP helpers
# --------------------------------------------------
def extract_entities(text):
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        if ent.label_ in TARGET_LABELS:
            entities.append((ent.text, ent.label_))

    return entities


def summarize_report(text):
    word_count = len(text.split())

    if word_count < 50:
        return text

    max_len = min(120, word_count // 2)
    min_len = max(20, min(50, max_len - 10))

    summary = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )

    return summary[0]["summary_text"]

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("AI Medical Report Analyzer")
st.caption(
    "Extracts key information and generates a simple summary. "
    "This tool does not provide medical diagnosis."
)

text = st.text_area(
    "Paste a medical report below:",
    height=220,
    placeholder="Paste the medical report text here..."
)

if st.button("Analyze Report"):
    if not text.strip():
        st.warning("Please paste a medical report to analyze.")
    else:
        with st.spinner("Analyzing report..."):
            entities = extract_entities(text)
            summary = summarize_report(text)

        # -------------------------
        # Entities Section
        # -------------------------
        st.subheader("Extracted Entities")

        if entities:
            grouped_entities = defaultdict(list)
            for ent, label in entities:
                grouped_entities[label].append(ent)

            for label, items in grouped_entities.items():
                st.markdown(f"**{label}**")
                st.write(", ".join(sorted(set(items))))
        else:
            st.write("No relevant entities detected.")

        # -------------------------
        # Summary Section
        # -------------------------
        st.subheader("Patient-Friendly Summary")
        st.write(summary)

        # -------------------------
        # Disclaimer
        # -------------------------
        st.info(DISCLAIMER)

