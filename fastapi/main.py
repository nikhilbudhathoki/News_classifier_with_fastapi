from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn
import os  # Add this import

# Define categories
CATEGORIES = [
    'Arts', 'Automobile', 'Bank', 'Blog', 'Business', 'Crime', 'Economy', 'Education',
    'Entertainment', 'Health', 'Politics', 'Society', 'Sports', 'Technology', 'Tourism', 'World'
]

app = FastAPI()

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
    
    # Tokenize input
    inputs = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
    predicted_class = CATEGORIES[predicted_class_idx]
    
    # Convert probabilities to dictionary
    all_probs = {CATEGORIES[i]: float(prob) for i, prob in enumerate(probabilities[0].cpu().numpy())}
    
    return {
        "predicted_category": predicted_class,
        "probabilities": all_probs
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use Koyeb's PORT env var, fallback to 8000 locally
    uvicorn.run(app, host="0.0.0.0", port=port)
