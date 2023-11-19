import streamlit as st
from PIL import Image
from model import load_blip_model, load_vit_gpt2_model, generate_blip_caption, generate_vit_gpt2_caption
from utils import load_image_from_url

def main():
    st.title("Image Captioning with AI")
    st.header('Upload an image and get a caption!')

    # Model selection
    model_choice = st.selectbox("Choose a captioning model:", ["VIT-GPT2", "BLIP"])

    # Load the selected model
    if model_choice == "VIT-GPT2":
        model, feature_extractor, tokenizer, device, gen_kwargs = load_vit_gpt2_model()
    else:
        model, processor, device = load_blip_model()

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    image_url = st.text_input("Enter or paste an image url here")
    submit_url = st.button("Load image from url")

    if uploaded_image is not None or image_url:
        if uploaded_image:
            image = Image.open(uploaded_image)
        elif submit_url:
            image = load_image_from_url(image_url) # I stopped here, remember!
            if image is None:
                return


        st.image(image, caption='Uploaded Image', use_column_width=True)

        if model_choice == "VIT-GPT2":
            # Generate caption using VIT-GPT2 model
            caption = generate_vit_gpt2_caption(image, model, feature_extractor, tokenizer, device, gen_kwargs)
        else:
            # Generate caption using BLIP model
            caption = generate_blip_caption(image, model, processor, device)

        st.write(f"Caption: {caption}")


if __name__ == '__main__':
    main()