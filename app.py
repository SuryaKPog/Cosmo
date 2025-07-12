from flask import Flask, request, jsonify, render_template
import json
from transformers import pipeline
from difflib import get_close_matches
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask with template_folder set to current directory
app = Flask(__name__, template_folder=os.path.dirname(os.path.abspath(__file__)))

# Load neptunebot.json with UTF-8 encoding
try:
    with open('neptunebot.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    logger.debug(f"Loaded neptunebot.json with {len(data)} entries")
except FileNotFoundError:
    print("Error: neptunebot.json not found in the project folder!")
    data = []
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON format in neptunebot.json - {e}")
    data = []
except UnicodeDecodeError as e:
    print(f"Error: Encoding issue in neptunebot.json - {e}")
    data = []

# Extract questions and answers, with safety check
questions = [item['question'].lower() for item in data if isinstance(item, dict) and 'question' in item]
context = " ".join([f"Q: {item['question']} A: {item['answer']}" for item in data if isinstance(item, dict) and 'question' in item and 'answer' in item])
logger.debug(f"Questions loaded: {questions[:5]}")  # Log first 5 questions
logger.debug(f"Context length: {len(context)} characters")

# Initialize DistilBERT for Q&A
try:
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    logger.debug("Transformer model loaded successfully")
except Exception as e:
    print(f"Error loading Transformer model: {e}")
    qa_pipeline = None

def find_answer(user_input):
    logger.debug(f"Processing user input: {user_input}")
    if not data:
        return "No space facts loaded! Check neptunebot.json."
    if not qa_pipeline:
        return "CosmosBot’s engines are down! Transformer model failed to load."

    # Step 1: Check for close matches in neptunebot.json
    user_input_lower = user_input.lower().strip()
    close_match = get_close_matches(user_input_lower, questions, n=1, cutoff=0.7)  # Increased cutoff for better accuracy
    if close_match:
        matched_question = close_match[0]
        logger.debug(f"Matched question: {matched_question}")
        for item in data:
            if item.get('question', '').lower() == matched_question:
                return f"Stellar query! {item['answer']}"
    
    # Step 2: Use Transformer for Q&A
    if context:
        try:
            result = qa_pipeline(question=user_input, context=context)
            answer = result['answer']
            score = result['score']
            logger.debug(f"Transformer answer: {answer}, score: {score}")
            if score > 0.4:  # Lowered threshold for flexibility
                return f"CosmosBot diving into the void: {answer}. Want more cosmic insights?"
        except Exception as e:
            logger.error(f"Transformer error: {e}")
    
    # Step 3: Fallback response
    logger.debug("No match found, using fallback")
    return "I’m lost in the cosmic haze! Try asking about galaxies, stars, or black holes."

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error loading index.html: {e}")
        return "Error: index.html not found in the project folder!"

@app.route('/chat', methods=['GET'])
def chat():
    user_input = request.args.get('message', '')
    if not user_input:
        return jsonify({'response': 'Ask me something about the universe!'})
    response = find_answer(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)