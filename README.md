Presentation - https://docs.google.com/presentation/d/162VPX6L8s2Ts7xf8eEu2CK8lmOK04zKl/edit?usp=share_link&ouid=106612026531666680892&rtpof=true&sd=true



# 🧠 Pneumonia Detection & AI Chat Assistant

This project is a complete **Pneumonia Detection and Recommendation System** that combines:

- 📷 A **Vision Transformer (ViT)**-based image classifier for detecting pneumonia from chest X-ray images.
- 🤖 A **fine-tuned Gemini chatbot**, trained on pneumonia-related medical literature, to assist users with detailed explanations and treatment advice.
- 🌐 An interactive **Streamlit** interface to provide real-time predictions and a conversational AI experience.

---

## 🔍 Features

- **ViT-based Image Classifier**:
  - Classifies chest X-ray images as either `PNEUMONIA` or `NORMAL`
  - Displays prediction confidence/probability
- **Chatbot Integration (Gemini)**:
  - Custom fine-tuned model for Pneumonia-related responses
  - Provides explanations, precautions, and treatments based on classification
- **Streamlit Web App**:
  - Upload image and get classification instantly
  - Automatically prompts the chatbot based on the result
  - Includes an interactive chat window for user queries

---

## 🧰 Tech Stack

- `Streamlit` for front-end web interface
- `Google Generative AI` (Gemini, fine-tuned model)
- `Transformers` by HuggingFace
- `torch` (PyTorch) for model loading/inference
- `ViT` (`vit-base-patch16-224-in21k`) pretrained model
- `FAISS`, `PIL`, `NumPy`, `scikit-learn` (optional for future expansion)

---

## 📂 Project Structure

.
├── vit_pneumonia_predictor.pth         # Fine-tuned ViT model weights
├── app.py                              # Streamlit app script
├── README.md                           # This file
└── requirements.txt                    # Python dependencies (to be added)

---

## 🚀 How to Run

### 1. Install Requirements

```bash
pip install streamlit torch torchvision transformers pillow google-generativeai

Make sure you have a valid API key from Google AI Studio for Generative AI.

2. Update the API Key

In the script, replace the placeholder:

genai.configure(api_key="YOUR_API_KEY_HERE")

3. Launch the App

streamlit run app.py



⸻

🧪 Model Info
	•	Model: ViT Base Patch16-224
	•	Dataset: Fine-tuned on pneumonia chest X-ray data
	•	Output: Binary classification (PNEUMONIA / NORMAL)

⸻

💬 Example Chat Interactions

You: What are common symptoms of pneumonia?
Bot: Pneumonia symptoms include chest pain, cough with phlegm, shortness of breath, and fever...

You: How is pneumonia treated?
Bot: Treatment depends on severity and type, but often includes antibiotics, oxygen therapy...



⸻

⚠️ Disclaimer

This tool is intended for educational and demonstrational purposes only. It is not a substitute for professional medical advice. Always consult a healthcare provider for diagnosis and treatment.


⸻

🙌 Acknowledgments
	•	Google Generative AI (Gemini)
	•	Hugging Face Transformers
	•	PyTorch Team
	•	Streamlit Community

