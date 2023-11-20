import streamlit as st
from PIL import Image
from model import load_blip_model, load_vit_gpt2_model, generate_blip_caption, generate_vit_gpt2_caption, load_stable_diffusion_model
from utils import load_image_from_url, generate_image_from_text
from ui import image_captioning_ui, text_to_image_ui

def main():
     # Custom CSS
    custom_css = """
    <style>
        /* Change the sidebar color */
        .sidebar .sidebar-content {
            background-color: blue;
        }

        /* Change the main area background color */
        body {
            background-color: white;
        }

        /* Change button color */
        .stButton>button {
            background-color: blue;
            color: white;
        }
    </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)

    st.title("Image to Text and Text to Image Application")

    app_mode = st.sidebar.selectbox("Choose the App Mode:",
                                    ["Image Captioning", "Text-to-Image Generation"])
    
    if app_mode == "Image Captioning":
        image_captioning_ui()

    elif app_mode == "Text-to-Image Generation":
        stable_diffusion_model = load_stable_diffusion_model()
        text_to_image_ui(stable_diffusion_model)


#     st.title("Image Captioning with AI")
#     st.header('Upload an image and get a caption!')

#     # Model selection
#     model_choice = st.selectbox("Choose a captioning model:", ["VIT-GPT2", "BLIP"])

#     # Load the selected model
#     if model_choice == "VIT-GPT2":
#         model, feature_extractor, tokenizer, device, gen_kwargs = load_vit_gpt2_model()
#     else:
#         model, processor, device = load_blip_model()

#     uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#     image_url = st.text_input("Enter or paste an image url here")
#     submit_url = st.button("Load image from url")

#     if uploaded_image is not None or image_url:
#         if uploaded_image:
#             image = Image.open(uploaded_image)
#         elif submit_url:
#             image = load_image_from_url(image_url) # I stopped coding here, remember!
#             if image is None:
#                 return


#         st.image(image, caption='Uploaded Image', use_column_width=True)

#         if model_choice == "VIT-GPT2":
#             # Generate caption using VIT-GPT2 model
#             caption = generate_vit_gpt2_caption(image, model, feature_extractor, tokenizer, device, gen_kwargs)
#         else:
#             # Generate caption using BLIP model
#             caption = generate_blip_caption(image, model, processor, device)

#         st.write(f"Caption: {caption}")


# # Text-to-Image Generation
#     st.header("Generate an Image from Text")
#     # Load the model for Text-to-Image
#     stable_diffusion_model = load_stable_diffusion_model()

#     input_text = st.text_input("Enter a description for image generation:")
#     generate_image_button = st.button("Generate Image")

#     if generate_image_button and input_text:
#         generated_image = generate_image_from_text(stable_diffusion_model, input_text)
#         st.image(generated_image, caption='Generated Image', use_column_width=True)


if __name__ == '__main__':
    main()