import streamlit as st
from transformers import AutoTokenizer
from transformers import PaliGemmaForConditionalGeneration
from PIL import Image
import io
import torch
from torchvision import transforms

# Set page title
st.title("Image Captioning App")

# Set up model and tokenizer
@st.cache_resource
def load_model():
    model_id = "harshad317/alt_text_long"
    
    # Load model
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, use_auth_token="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",
        device_map="auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token="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")

    return model, tokenizer

model, tokenizer = load_model()

# Function to generate caption
def generate_caption(input_text, input_image):
    try:
        # Prepare input
        image = input_image.convert('RGB')
        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
        
        # Process image
        pixel_values = preprocess_image(image).unsqueeze(0).to(model.device)

        # Generate caption
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pixel_values=pixel_values,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return None

# Helper function to preprocess the image
def preprocess_image(image):
    # Implement image preprocessing here
    # This might include resizing, normalization, etc.
    # Return a tensor of the preprocessed image
    preprocess = transforms.Compose([
        transforms.Resize((1024, 1024)),  # Resize to 1024x1024
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)

# Create input fields
input_text = st.text_input("Enter a long description for the given image")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Add error handling and debugging statements
if uploaded_image is not None:
    try:
        input_image = Image.open(io.BytesIO(uploaded_image.read()))
        st.write("Image uploaded successfully!")
    except Exception as e:
        st.error(f"Error uploading image: {e}")
        input_image = None
else:
    st.write("No image uploaded")
    input_image = None

# Button to generate caption
if st.button("Generate Caption"):
    if input_image is not None:
        caption = generate_caption(input_text, input_image)
        if caption is not None:
            st.write("Generated Caption:")
            st.write(caption)
    else:
        st.write("Please upload an image")