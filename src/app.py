import streamlit as st
from predict import SpamDetector
st.set_page_config(
    page_title="Spam Detection",
    page_icon="📧",
    layout="centered"
)
detector = SpamDetector()
