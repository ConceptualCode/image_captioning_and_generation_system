from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
import torch


def load_vit_gpt2_model():
    # Load the model and its components
    model_name="nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Generation arguments
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
    # Convert the PIL image to the format expected by the model
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    # Process the image with the feature extractor
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate the caption using the model
    output_ids = model.generate(pixel_values, **gen_kwargs)

    # Decode the output IDs to a caption
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return caption


def generate_vit_gpt2_caption(image, model, feature_extractor, tokenizer, device, gen_kwargs):
    # Convert the PIL image to the format expected by the model
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    # Process the image with the feature extractor
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate the caption using the model
    output_ids = model.generate(pixel_values, **gen_kwargs)

    # Decode the output IDs to a caption
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return caption

def generate_blip_caption(image, model, processor, device):
    # Convert the PIL image to the format expected by the model
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    # Process the image with the processor
    inputs = processor(image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate the caption using the model
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    # Generate the caption using the model
    unwanted_prefix = "arafed"
    if caption.startswith(unwanted_prefix):
        caption = caption[len(unwanted_prefix):].lstrip()

    return caption
