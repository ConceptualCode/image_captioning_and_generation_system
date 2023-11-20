import streamlit as st
from model import load_stable_diffusion_model, load_blip_model, load_vit_gpt2_model, generate_vit_gpt2_caption, generate_blip_caption
from utils import generate_image_from_text, load_image_from_url
from PIL import Image


def image_captioning_ui():
    st.header("Image Captioning")

    caption_model_choice = st.selectbox("Choose an image captioning model:", ["VIT-GPT2", "BLIP"])

    if caption_model_choice == "VIT-GPT2":
        model, feature_extractor, tokenizer, device, gen_kwargs = load_vit_gpt2_model()
    else:
        model, processor, device = load_blip_model()

    uploaded_image = st.file_uploader("Upload an image for captioning", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("Enter or paste an image URL here")
    submit_url = st.button("Load Image from URL")

    image = None

    if uploaded_image is not None or (submit_url and image_url):
        with st.spinner("Processing image..."):
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
            elif submit_url and image_url:
                image = load_image_from_url(image_url)

            if image:
                st.image(image, caption='Uploaded Image', use_column_width=True)

                if caption_model_choice == "VIT-GPT2":
                    caption = generate_vit_gpt2_caption(image, model, feature_extractor, tokenizer, device, gen_kwargs)
                else:
                    caption = generate_blip_caption(image, model, processor, device)
                
                st.write(f"Caption: {caption}")
            else:
                st.warning("Please upload an image or enter a valid image URL.")

# def image_captioning_ui():
#     st.header("Image Captioning")

#     # Model selection for Image Captioning
#     caption_model_choice = st.selectbox("Choose an image captioning model:", ["VIT-GPT2", "BLIP"])

#     # Load the selected model
#     if caption_model_choice == "VIT-GPT2":
#         model, feature_extractor, tokenizer, device, gen_kwargs = load_vit_gpt2_model()
#     else:
#         model, processor, device = load_blip_model()

#     uploaded_image = st.file_uploader("Upload an image for captioning", type=["jpg", "jpeg", "png"])

#     image_url = st.text_input("Enter or paste an image url here")
#     submit_url = st.button("Load image from url")

#     if uploaded_image is not None or image_url:
#         with st.spinner("Processing image..."):
#             if uploaded_image:
#                 image = Image.open(uploaded_image)
#             elif submit_url:
#                 image = load_image_from_url(image_url) # I stopped coding here, remember!
#                 if image is None:
#                     return
                
#             st.image(image, caption='Uploaded Image', use_column_width=True)

#             if caption_model_choice == "VIT-GPT2":
#                 caption = generate_vit_gpt2_caption(image, model, feature_extractor, tokenizer, device, gen_kwargs)
#             else:
#                 caption = generate_blip_caption(image, model, processor, device)
            
#             st.write(f"Caption: {caption}")


def text_to_image_ui(stable_diffusion_model):
    st.header("Text-to-Image Generation")
    input_text = st.text_input("Enter a description for image generation:")

    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            generated_image = generate_image_from_text(stable_diffusion_model, input_text)
            st.image(generated_image, caption='Generated Image', use_column_width=True)