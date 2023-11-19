# Image to Text and Text to Image Application


## Overview

This Streamlit-based application leverages cutting-edge AI models to offer two main functionalities: Image to Text and Text-to-Image Generation. Users can select between these two modes to either generate descriptive captions for images or create images from textual descriptions.

## Features

1. **Image Captioning**: 
   - Upload an image or provide an image URL.
   - Choose between two captioning models: VIT-GPT2 and BLIP.
   - Generate and view image captions.

2. **Text-to-Image Generation**:
   - Enter a descriptive text.
   - Generate and view images created from the text.

## Installation

To run this app, you'll need Python installed on your system, along with the Streamlit library and other dependencies.

1. **Clone the Repository:**
   ```
   git clone https://github.com/ConceptualCode/image_captioning_and_generation_system.git
   ```

2. **Navigate to the App Directory:**
   ```
   cd image_captioning_and_generation_system
   ```

3. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the App:**
   ```
   streamlit run app.py
   ```

## Usage

After starting the app, navigate through the sidebar to select between the Image Captioning and Text-to-Image Generation features.

- For **Image Captioning**:
  - Upload an image or enter an image URL.
  - Select the captioning model.
  - Click 'Process Image' to view the caption.

- For **Text-to-Image Generation**:
  - Enter your description in the text box.
  - Click 'Generate Image' to view the generated image.

## Technologies Used

- Streamlit
- Latent Stable Diffusion
- Text to Image
- Image Captioning
- PyTorch
- Transformers
- PIL (Python Imaging Library)

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

[MIT License](LICENSE)

## Contact

For any queries or suggestions, please contact [Anthony Soronnadi](anthony12soronnadi@gmail.com).

---

### References:

- [CompVis](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- [NLPConnect](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)
- [Salesforce](https://huggingface.co/Salesforce/blip-image-captioning-large)