import streamlit as st
from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from PIL import Image
import requests
import torch

# Set up model and processor
model_id_short = "harshad317/alt_text_short"  # Replace with your actual model ID
model_short = PaliGemmaForConditionalGeneration.from_pretrained(model_id_short, torch_dtype=torch.bfloat16, use_auth_token = "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")
processor_short = PaliGemmaProcessor.from_pretrained(model_id_short, use_auth_token = "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")

# Set up device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_short.to(device)

def generate_description(image_url):
    input_text_short = "Write short description for the given image"
    input_image = Image.open(requests.get(image_url, stream=True).raw)

    inputs = processor_short(text=input_text_short, images=input_image,
                              padding="longest", do_convert_rgb=True, return_tensors="pt").to(device)
    inputs = inputs.to(dtype=model_short.dtype)

    with torch.no_grad():
        output = model_short.generate(**inputs, max_length=2048)
    return processor_short.decode(output[0], skip_special_tokens=True)

# Create Streamlit app
st.title("Image Description Generator")

# Input field for image URL
image_url = st.text_input("Enter image URL", value="https://huggingface.co/datasets/ritwikraha/random-storage/resolve/main/cohen.jpeg")

# Button to generate description
if st.button("Generate Description"):
    description = generate_description(image_url)
    st.write("Generated Description:")
    st.write(description)