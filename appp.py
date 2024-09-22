import os
import streamlit as st
import google.generativeai as genai
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import numpy as np

# Google Generative AI Configuration
genai.configure(api_key="AIzaSyDC7Th_hRNzyEPVWglDy5sLA43qcmAalVA")  

# Chat Model Configuration
generation_config = {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="tunedModels/chatpneumonia-x7oggdrfeasj",
    generation_config=generation_config,
)

chat_session = model.start_chat(history=[])



# Load ViT Model
model_name_or_path = 'google/vit-base-patch16-224-in21k'
vit_feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

vi_model = ViTForImageClassification.from_pretrained(model_name_or_path, num_labels=2)
vi_model.load_state_dict(torch.load("/Users/rishvanthgv/Documents/hackfest_gfg/vit_pneumonia_predictor.pth", map_location=torch.device('cpu')))
vi_model.eval()

label_map = {0: 'PNEUMONIA', 1: 'NORMAL'}

def predict_image_class(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = vit_feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = vi_model(pixel_values=inputs['pixel_values'])
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        probs_np = probs.numpy()
        predicted_class = torch.argmax(probs, dim=1).item()
        predicted_label = label_map[predicted_class]

        # Adding the conditional printing logic
        if predicted_label == "PNEUMONIA":
            x = probs_np[0, 0] * 100
        elif predicted_label == "NORMAL":
            x = probs_np[0, 1] * 100

    return predicted_label, probs_np, x



# Streamlit UI
st.title("Complete Pneumonia Assistance")
st.markdown(
    """
    <style>
    .title {
        color: #FF0000;  /* Change to your desired color (e.g., a shade of red) */
    }
    /* Remove padding and margin from the main container */
    .block-container {
        padding: 80;
        margin: 30;
        max-width: 100%;
    }

    /* Remove padding from the main area */
    .main {
        padding: 0;
    }

    /* Adjust column spacing to eliminate margins */
    [class^="css-1lcbmhc"] {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create two wider columns
col1, col2, col3 = st.columns([1,3,1])  # Adjust the ratio to control column width and gap


disease = ""
with col1:
    st.header("Prediction Model")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")
        predicted_label, probabilities, y = predict_image_class(uploaded_file)
        disease = predicted_label
        st.write(f"Predicted class: {predicted_label}")
        st.write(f"Probability of {predicted_label}: {y} %")
        # st.write(f"Probabilities: {probabilities}")

with col2:
    st.header("Report")
    if disease == "NORMAL":
        response = chat_session.send_message("What is Pneumonia ?")
        st.write("\n".join(response.text.split( "\n")[1:]))
    if disease == "PNEUMONIA":
        response = chat_session.send_message("What are the treatments for Pneumonia ?")
        st.write("\n".join(response.text.split( "\n")[1:]))

with col3:
    st.markdown('<div class="column-border">', unsafe_allow_html=True)
    st.header("Chat With AI")
    if 'history' not in st.session_state:
        st.session_state['history'] = []  # Track conversation history

    # User input area
    user_input = st.text_input("You:", "", key="input")

    # If user presses enter and input is not empty
    if user_input:
        # Send the user's input to the model
        response = chat_session.send_message(user_input)
        st.session_state['history'].append({"user": user_input, "bot": response.text})

    # Display chat history in reverse order (latest at the top)
    if st.session_state['history']:
        for chat in reversed(st.session_state['history']):
            st.write(f"You: {chat['user']}")
            st.write(f"Assistant: {chat['bot']}")
    
    

