from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
import os

# Define categories
CATEGORIES = [
    'Arts', 'Automobile', 'Bank', 'Blog', 'Business', 'Crime', 'Economy', 'Education',
    'Entertainment', 'Health', 'Politics', 'Society', 'Sports', 'Technology', 'Tourism', 'World'
]

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model, tokenizer, and device
model = None
tokenizer = None
device = None

# Load model at startup
@app.on_event("startup")
async def load_model():
    global model, tokenizer, device
    model_name = "NikhilBudhathoki/News_classifier"
    try:
        print(f"Loading model from Hugging Face: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = 'cpu'  # Koyeb free tier is CPU-only
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Pydantic model for input
class TextInput(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict_category(input: TextInput):
    if len(input.text.strip()) < 10:
        return {"error": "Text too short, please provide longer input"}
    
    inputs = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
    predicted_class = CATEGORIES[predicted_class_idx]
    
    all_probs = {CATEGORIES[i]: float(prob) for i, prob in enumerate(probabilities[0].cpu().numpy())}
    
    return {
        "predicted_category": predicted_class,
        "probabilities": all_probs
    }

# Optional root endpoint (explains 200 OK for GET /)
@app.get("/")
def read_root():
    return {"message": "FastAPI News Classifier is running"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Koyeb sets PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
