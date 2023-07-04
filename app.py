import streamlit as st
import re
from transformers import pipeline

# Set up Streamlit
st.title("Article Summarizer")

# Function to preprocess and summarize the article
def summarize_article(article_text):
    # Text preprocessing
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    # Use T5 model for summarization
    summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
    summary = summarizer(formatted_article_text, min_length=90, max_length=int(len(article_text)/2), do_sample=False)

    return summary[0]['summary_text']

# Streamlit app
def main():
    # Get user input
    article_text = st.text_area("Enter the article text:", height=200)

    if st.button("Summarize"):
        if article_text:
            # Summarize the article
            summary = summarize_article(article_text)

            # Display the summary
            st.header("Summary")
            st.write(summary)
        else:
            st.warning("Please enter the article text.")

if __name__ == "__main__":
    main()