import streamlit as st
import spacy
from transformers import pipeline
from collections import defaultdict
import re

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="AI Medical Report Analyzer",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# Load models (cached)
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
# Text normalization (CRITICAL FIX)
# --------------------------------------------------
def normalize_text(text):
    """
    Fix missing spaces from PDFs / OCR / bad copy-paste.
    """
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([.,;:])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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


def summarize_large_report(text):
    if not text or len(text.split()) < 80:
        return text

    summaries = []

    tokenizer = summarizer.tokenizer
    max_input_tokens = 900  # well below BART limit (1024)

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=False
    )["input_ids"][0]

    for i in range(0, len(tokens), max_input_tokens):
        chunk_ids = tokens[i:i + max_input_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        try:
            result = summarizer(
                chunk_text,
                max_length=120,
                min_length=40,
                do_sample=False,
                truncation=True
            )
            summaries.append(result[0]["summary_text"])
        except Exception:
            continue  # fail-safe

    if not summaries:
        return "Unable to safely summarize this report due to formatting or length."

    combined_summary = " ".join(summaries)

    # Optional final compression
    if len(combined_summary.split()) > 160:
        try:
            final = summarizer(
                combined_summary,
                max_length=120,
                min_length=50,
                do_sample=False,
                truncation=True
            )
            return final[0]["summary_text"]
        except Exception:
            return combined_summary

    return combined_summary

# --------------------------------------------------
# UI
# --------------------------------------------------
st.markdown("""
<div style="text-align:center; padding:30px 10px;">
    <h1>🧠 AI Medical Report Analyzer</h1>
    <p style="color:#b0b0b0; font-size:16px;">
        Extracts key information and generates a patient-friendly summary
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Large medical reports are automatically cleaned and summarized in parts "
    "to ensure accuracy and stability."
)

st.markdown("---")

raw_text = st.text_area(
    "Paste a medical report below",
    height=280,
    placeholder="You can paste long lab reports, discharge summaries, legal or diagnostic notes..."
)

if st.button("Analyze Report"):
    if not raw_text.strip():
        st.warning("Please paste a medical report to analyze.")
    else:
        with st.spinner("Analyzing report (this may take a moment for large documents)..."):
            cleaned_text = normalize_text(raw_text)
            entities = extract_entities(cleaned_text)
            summary = summarize_large_report(cleaned_text)

        # -------------------------
        # Entities Section
        # -------------------------
        st.subheader("📌 Extracted Key Information")

        if entities:
            grouped_entities = defaultdict(set)
            for ent, label in entities:
                grouped_entities[label].add(ent)

            for label, items in grouped_entities.items():
                st.markdown(f"**{label}**")
                st.write(", ".join(sorted(items)))
        else:
            st.write("No relevant entities detected.")

        # -------------------------
        # Summary Section
        # -------------------------
        st.subheader("📝 Patient-Friendly Summary")
        st.markdown(
            f"""
            <div style="
                background-color:#ffffff;
                padding:20px;
                border-radius:12px;
                color:#1a1a1a;
                box-shadow:0 6px 16px rgba(0,0,0,0.15);
            ">
                {summary}
            </div>
            """,
            unsafe_allow_html=True
        )

        # -------------------------
        # Disclaimer
        # -------------------------
        st.info(DISCLAIMER)

        # -------------------------
        # Debug (optional)
        # -------------------------
        with st.expander("🔍 View cleaned input text"):
            st.write(cleaned_text)
