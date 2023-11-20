from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
import torch
from diffusers import StableDiffusionPipeline

from dotenv import load_dotenv
import os

load_dotenv()

auth_token = os.getenv("HUGGINGFACE_TOKEN")

def load_vit_gpt2_model():
    model_name="nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    gen_kwargs = {"max_length": 16, "num_beams": 4}

    return model, feature_extractor, tokenizer, device, gen_kwargs

def load_blip_model():
    model_name = "Salesforce/blip-image-captioning-large"
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    processor = BlipProcessor.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, processor, device


def generate_caption(image, model, feature_extractor, tokenizer, device, gen_kwargs):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return caption


def generate_vit_gpt2_caption(image, model, feature_extractor, tokenizer, device, gen_kwargs):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return caption

def generate_blip_caption(image, model, processor, device):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    inputs = processor(image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    unwanted_prefix = "arafed"
    if caption.startswith(unwanted_prefix):
        caption = caption[len(unwanted_prefix):].lstrip()

    return caption


def load_stable_diffusion_model(model_id="CompVis/stable-diffusion-v1-4", device="cuda"):
    auth_token = os.getenv("HUGGINGFACE_TOKEN")
    model = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
    model.to(device)
    return model

