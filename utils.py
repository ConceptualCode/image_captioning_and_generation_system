import requests
import streamlit as st
from PIL import Image
#from transformers import StableDiffusionPipeline
import torch

def load_image_from_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return Image.open(response.raw).convert('RGB')
    except requests.RequestException as e:
        st.error(f"Error loading image from URL: {e}")
        return None

def generate_image_from_text(model, prompt, device="cuda"):
    with torch.no_grad():
        generated_images = model(prompt)["images"]
        
    generated_image = generated_images[0]

    return generated_image