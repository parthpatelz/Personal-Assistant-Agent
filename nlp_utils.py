from transformers import pipeline

# Load the NLP model
nlp = pipeline("text-classification", model="distilbert-base-uncased")

def process_input(text):
    # Use the model to classify the input text
    result = nlp(text)
    intent = result[0]['label']
    confidence = result[0]['score']
    return f"Understood your intent: {intent} (Confidence: {confidence:.2f})"