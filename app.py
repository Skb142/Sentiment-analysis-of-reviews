import streamlit as st
import requests

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter a product review below and get its sentiment prediction.")

# User Input
review_text = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if review_text:
        # Send the review to the Flask API
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"review": review_text}
        )

        if response.status_code == 200:
            result = response.json().get("prediction", "Error: No prediction returned").lower()

            # Show result with color formatting
            if result == "positive":
                st.success(f"✅ Sentiment: **{result.capitalize()}**", icon="✅")
                st.markdown(f'<div style="background-color:#d4edda;color:#155724;padding:10px;border-radius:5px;">'
                            f'<b>Sentiment:</b> {result.capitalize()}</div>', unsafe_allow_html=True)
            elif result == "negative":
                st.error(f"❌ Sentiment: **{result.capitalize()}**", icon="❌")
                st.markdown(f'<div style="background-color:#f8d7da;color:#721c24;padding:10px;border-radius:5px;">'
                            f'<b>Sentiment:</b> {result.capitalize()}</div>', unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Sentiment: **{result.capitalize()}**")

        else:
            st.error("Error in API request. Please check the backend.")
    else:
        st.warning("Please enter a review before submitting.")
