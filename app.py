import streamlit as st
import spacy
from transformers import pipeline
from collections import defaultdict
import math

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

MAX_WORDS_PER_CHUNK = 400   # safe for BART
SUMMARY_MAX_LEN = 120
SUMMARY_MIN_LEN = 40

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


def split_text_into_chunks(text, max_words=MAX_WORDS_PER_CHUNK):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)

    return chunks


def summarize_large_report(text):
    if not text or len(text.split()) < 80:
        return text

    summaries = []

    # Tokenizer-aware chunking
    max_input_tokens = 900  # keep well below 1024
    tokenizer = summarizer.tokenizer

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=False
    )

    input_ids = inputs["input_ids"][0]

    for i in range(0, len(input_ids), max_input_tokens):
        chunk_ids = input_ids[i:i + max_input_tokens]

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True
        )

        try:
            summary = summarizer(
                chunk_text,
                max_length=120,
                min_length=40,
                do_sample=False,
                truncation=True
            )
            summaries.append(summary[0]["summary_text"])
        except Exception:
            # Fail-safe: skip bad chunk instead of crashing
            continue

    if not summaries:
        return "Unable to summarize this report safely due to length or formatting."

    combined = " ".join(summaries)

    # Optional final compression
    if len(combined.split()) > 160:
        try:
            final = summarizer(
                combined,
                max_length=120,
                min_length=50,
                do_sample=False,
                truncation=True
            )
            return final[0]["summary_text"]
        except Exception:
            return combined

    return combined

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

st.markdown("---")

text = st.text_area(
    "Paste a medical report below",
    height=260,
    placeholder="You can paste long lab reports, discharge summaries, or diagnostic notes..."
)

if st.button("Analyze Report"):
    if not text.strip():
        st.warning("Please paste a medical report to analyze.")
    else:
        with st.spinner("Analyzing report (large documents may take a moment)..."):
            entities = extract_entities(text)
            summary = summarize_large_report(text)

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
