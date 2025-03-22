# # app.py
# import spacy

# # Load the trained intent classifier
# nlp = spacy.load("intent_classifier")

# # Add EntityRuler to detect tasks/times
# ruler = nlp.add_pipe("entity_ruler", after="textcat")
# patterns = [
#     {"label": "TASK", "pattern": [{"LOWER": "call"}, {"LOWER": "mom"}]},
#     {"label": "TIME", "pattern": [{"LOWER": "at"}, {"SHAPE": "d"}, {"LOWER": "pm"}]},
# ]
# ruler.add_patterns(patterns)

from flask import Flask, request, jsonify
from nlp_utils import process_input

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"  

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response = process_input(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)