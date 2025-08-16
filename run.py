
# Simple LLM Application with LangChain Expression Language (LCEL)

# 

from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize Groq model
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Prompt template
generic_template = "Translate the following into {language}:"
prompt = ChatPromptTemplate.from_messages([
    ("system", generic_template),
    ("user", "{text}")
])

# Output parser
parser = StrOutputParser()

# Chain components
chain = prompt | model | parser

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Translation API! Use POST /translate with JSON body."

@app.route("/translate", methods=["POST"])
def translate():
    # Expect JSON: { "text": "Hello", "language": "French" }
    data = request.get_json()
    
    if not data or "text" not in data or "language" not in data:
        return jsonify({"error": "POST JSON must contain 'text' and 'language'"}), 400
    
    text = data["text"]
    language = data["language"]
    
    result = chain.invoke({"language": language, "text": text})
    return jsonify({"original": text, "translation": result, "language": language})
 

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8001, debug=True)

