
import os
import google.generativeai as genai

genai.configure(api_key="API_KEY")

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="tunedModels/chatpneumonia-x7oggdrfeasj",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

chat_session = model.start_chat(
  history=[
  ]
)

# Infinite loop to keep asking questions until 'exit' is entered
def chat_help():
  while True:
      user_input = input("You: ")

      if user_input.lower() == "exit":
          print("Exiting chat.")
          break

      response = chat_session.send_message(user_input)
      print(response.text)
chat_help()


# response = chat_session.send_message("can bacteria cause pneumonia")

# print(response.text)
