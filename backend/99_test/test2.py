from langchain.chat_models import init_chat_model

model = init_chat_model(
    model = "gemini-3-flash-preview",
    # base_url = "None",
    model_provider = "google_genai",
    api_key = "AIzaSyCP_qZBGWZzdRXIfiR3Fa7PpaJicbYIVZk",
)

response = model.invoke("Hello, how are you?")

print(response.content)