import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import numpy as np
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

model_name_or_path = 'google/vit-base-patch16-224-in21k'
vit_feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

model = ViTForImageClassification.from_pretrained(model_name_or_path, num_labels=2)
model.load_state_dict(torch.load("/Users/rishvanthgv/Documents/hackfest_gfg/vit_pneumonia_predictor.pth", map_location=torch.device('cpu')))
model.eval()

label_map = {0: 'PNEUMONIA', 1: 'NORMAL'}

def predict_image_class(image_path, model, feature_extractor):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(pixel_values=inputs['pixel_values'])
        
        # Get the logits and apply softmax to get probabilities
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        
        # Convert probabilities to numpy array
        probs_np = probs.numpy()

        # Get the predicted class
        predicted_class = torch.argmax(probs, dim=1).item()
        predicted_label = label_map[predicted_class]

        # Adding the conditional printing logic
        if predicted_label == "PNEUMONIA":
            # print("Pneumonia probability: ")
            x = probs_np[0, 0] * 100
        elif predicted_label == "NORMAL":
            # print("Normal probability: ")
            x = probs_np[0, 1] * 100

    return predicted_label, probs_np, x

# Test the function with a sample image
def disp(image_path):
    predicted_label, probabilities, y = predict_image_class(image_path, model, vit_feature_extractor)

    print(f"The predicted class for the image is: {predicted_label}")
    print(f"Probabilities for each class: {probabilities}")
    print(f"The probability of {predicted_label} is {y} %")

disp("download.jpg")
