import streamlit as st
from predict_sentiment import predict_sentiment

st.set_page_config(page_title="Sentiment Classifier", page_icon="💬")
st.title("📦 E-commerce Review Sentiment Classifier")
st.markdown("Enter your product review and get the sentiment prediction:")

user_input = st.text_area("Write your review here:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        try:
            sentiment, confidence = predict_sentiment(user_input)
            st.success(f"🧠 Sentiment: **{sentiment}** with confidence **{confidence:.2f}%**")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a review to analyze.")
