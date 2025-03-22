from transformers import pipeline
nlp = pipeline("text-classification", model="distilbert-base-uncased")