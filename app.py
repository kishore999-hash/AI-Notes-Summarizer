import streamlit as st
from transformers import pipeline
import pdfplumber

# Page config
st.set_page_config(page_title="AI Notes Summarizer", page_icon="🧠", layout="wide")

st.title("🧠 AI-Powered Notes Summarizer")
st.write("Upload a text file or paste your notes below to generate a concise summary!")

# Load model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Input method
option = st.radio("Choose Input Type:", ["Paste Text", "Upload PDF"])

text_input = ""

if option == "Paste Text":
    text_input = st.text_area("Paste your notes or article here:", height=250)
else:
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file:
        with pdfplumber.open(pdf_file) as pdf:
            text_input = ""
            for page in pdf.pages:
                text_input += page.extract_text() + "\n"

if text_input:
    summary_mode = st.selectbox("Select Summary Type", ["Short Summary", "Bullet Points"])
    if st.button("Generate Summary"):
        st.info("Summarizing... please wait ⏳")
        # Limit long text for model efficiency
        text_chunks = [text_input[i:i+1000] for i in range(0, len(text_input), 1000)]
        summarized_text = ""

        for chunk in text_chunks:
            result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summarized_text += result[0]['summary_text'] + " "

        if summary_mode == "Bullet Points":
            bullets = summarized_text.split(". ")
            summarized_text = "\n".join([f"• {b.strip()}" for b in bullets if b.strip()])

        st.subheader("📝 Summary:")
        st.success(summarized_text.strip())
