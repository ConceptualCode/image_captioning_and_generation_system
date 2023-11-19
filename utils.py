import requests
import streamlit as st
from PIL import Image

def load_image_from_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return Image.open(response.raw).convert('RGB')
    except requests.RequestException as e:
        st.error(f"Error loading image from URL: {e}")
        return None