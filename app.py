import streamlit as st

from predict import SpamDetector

# ==========================
# Load model
# ==========================

@st.cache_resource
def load_model():
    return SpamDetector()

detector = load_model()

# ==========================
# Giao diện
# ==========================

st.set_page_config(
    page_title="Spam Detection",
    page_icon="📧"
)

st.title("📧 Spam Detection App")

st.write(
    "Nhập một đoạn email hoặc tin nhắn bằng tiếng Anh để kiểm tra xem nó là **Spam** hay **Ham**."
)

text = st.text_area(
    "Enter your message",
    height=180
)

# ==========================
# Predict
# ==========================

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter a message.")
    else:

        label, confidence = detector.predict_probability(text)

        if label.lower() == "spam":

            st.error("🚨 Spam")

        else:

            st.success("✅ Ham")

        st.write(f"**Prediction:** {label}")

        st.write(f"**Confidence:** {confidence:.2%}")
