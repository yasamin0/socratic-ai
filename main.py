from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, FileResponse
from openai import OpenAI
import spacy
import os
import pickle
from datetime import datetime
import subprocess

# --------------------------------------
# Load spaCy NLP model (en_core_web_sm)
# If not available, download it on the fly
# --------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# --------------------------------------
# Load trained ML model and vectorizer
# Used for categorizing user input
# --------------------------------------
with open("model.pkl", "rb") as f:
    ml_model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    ml_vectorizer = pickle.load(f)

# --------------------------------------
# Initialize OpenAI client using environment variables
# Requires valid OPENAI_API_KEY and OPENAI_PROJECT_ID
# --------------------------------------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT_ID")
)

# --------------------------------------
# Chat history to track dialogue
# Starts with a system prompt setting the Socratic tone
# --------------------------------------
conversation_history = [
    {
        "role": "system",
        "content": "You are a Socratic philosopher. Always reply with thoughtful questions."
    }
]

# Create FastAPI app
app = FastAPI()

# --------------------------------------
# Categorize input using pre-trained ML model
# --------------------------------------
def categorize_input(text):
    X = ml_vectorizer.transform([text])
    return ml_model.predict(X)[0]

# --------------------------------------
# Preprocess input text (lemmatization, stopword removal)
# --------------------------------------
def preprocess(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

# --------------------------------------
# Home route - renders the chat UI
# --------------------------------------
@app.get("/", response_class=HTMLResponse)
async def form_page():
    return render_chat()

# --------------------------------------
# Handle user question submission
# --------------------------------------
@app.post("/ask", response_class=HTMLResponse)
async def ask_question(question: str = Form(...)):
    try:
        # Log user's question with timestamp
        conversation_history.append({
            "role": "user",
            "content": question,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Preprocess and categorize the question
        clean_question = preprocess(question)
        category = categorize_input(question)

        # Generate assistant's response from OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Socratic philosopher. Normally respond with thoughtful questions, "
                        "but if the user directly asks for your perspective or explanation, "
                        "provide a brief philosophical insight first, then follow up with a question."
                    )
                }
            ] + conversation_history[1:]
        )

        # Extract and log the assistant's answer
        answer = response.choices[0].message.content
        conversation_history.append({
            "role": "assistant",
            "content": answer,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        return render_chat(extra_info=f"Category: {category}")
    except Exception as e:
        return render_chat(error=f"Unexpected error: {str(e)}")

# --------------------------------------
# Reset chat history (clear previous conversation)
# --------------------------------------
@app.post("/reset", response_class=HTMLResponse)
async def reset_chat():
    conversation_history.clear()
    conversation_history.append({
        "role": "system",
        "content": "You are a Socratic philosopher. Always reply with thoughtful questions."
    })
    return render_chat()

# --------------------------------------
# Download current chat history as a text file
# --------------------------------------
@app.get("/download")
async def download_chat():
    filename = "chat_history.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for msg in conversation_history[1:]:
                timestamp = msg.get("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                sender = msg["role"].upper()
                content = msg["content"]
                f.write(f"[{timestamp}] {sender}: {content}\n\n")

        if os.path.exists(filename):
            return FileResponse(path=filename, media_type='text/plain', filename=filename)
        else:
            return {"error": "File not created"}
    except Exception as e:
        return {"error": f"Download failed: {str(e)}"}

# --------------------------------------
# Helper function to render the chat interface (HTML)
# --------------------------------------
def render_chat(error="", extra_info=""):
    chat_html = ""
    for msg in conversation_history[1:]:
        time_html = f"<div class='small text-muted'>{msg.get('time', '')}</div>"
        if msg["role"] == "user":
            chat_html += f"""
            <div class='text-end'>
                <div class='d-inline-block bg-primary text-white p-2 rounded mb-2 text-break'
                     style='max-width: 75%; min-width: 60px; padding: 0.5rem 1rem; word-break: break-word;'>
                    {msg['content']}
                    {time_html}
                </div>
            </div>
            """
        elif msg["role"] == "assistant":
            chat_html += f"""
            <div class='text-start'>
                <div class='d-inline-block bg-light text-dark p-2 rounded mb-2 text-break'
                     style='max-width: 75%; min-width: 60px; padding: 0.5rem 1rem; word-break: break-word;'>
                    {msg['content']}
                    {time_html}
                </div>
            </div>
            """

    return f"""
    <html>
        <head>
            <title>Socratic Chat</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
            <style>
                body {{ background-color: #f8f9fa; }}
                .chat-box {{
                    max-height: 70vh;
                    overflow-y: auto;
                    padding: 1rem;
                    border: 1px solid #ddd;
                    background: white;
                    border-radius: 8px;
                }}
                input[type="text"]::placeholder {{ font-style: italic; }}
                .small.text-muted {{ font-size: 0.75rem; margin-top: 0.25rem; }}
            </style>
        </head>
        <body class="container mt-5">
            <h2 class="mb-3">Socratic Dialogue</h2>

            <!-- Question Form -->
            <form action="/ask" method="post" class="mb-3">
                <div class="input-group">
                    <input type="text" class="form-control" name="question" placeholder="Type your question..." required>
                    <button type="submit" class="btn btn-primary">Send</button>
                </div>
            </form>

            <!-- Reset and Download Options -->
            <form action="/reset" method="post" class="mb-4">
                <button type="submit" class="btn btn-outline-danger btn-sm">🧹 Reset Chat</button>
            </form>
            <form action="/download" method="get" class="mb-3 d-inline">
                <button type="submit" class="btn btn-outline-secondary btn-sm">⬇️ Download Chat</button>
            </form>

            <!-- Extra Info or Error -->
            {f"<div class='alert alert-success'><strong>{extra_info}</strong></div>" if extra_info else ""}
            <div class="chat-box mb-4">{chat_html}</div>
            {"<div class='alert alert-danger mt-3'>" + error + "</div>" if error else ""}
        </body>
    </html>
    """

# --------------------------------------
# Route: /reset (POST)
# Description: Resets the conversation history to its initial state.
# This clears all prior messages and re-initializes the system prompt.
# --------------------------------------
@app.post("/reset", response_class=HTMLResponse)
async def reset_chat():
    conversation_history.clear()  # Remove all chat history
    conversation_history.append({  # Re-add the system message
        "role": "system",
        "content": "You are a Socratic philosopher. Always reply with thoughtful questions."
    })
    return render_chat()  # Return the refreshed chat interface

# --------------------------------------
# Route: /download (GET)
# Description: Exports the chat history as a .txt file.
# Each message includes a timestamp, sender role, and content.
# --------------------------------------
@app.get("/download")
async def download_chat():
    filename = "chat_history.txt"
    try:
        # Write each message to file with formatted timestamp
        with open(filename, "w", encoding="utf-8") as f:
            for msg in conversation_history[1:]:  # Skip the initial system message
                timestamp = msg.get("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                sender = msg["role"].upper()
                content = msg["content"]
                f.write(f"[{timestamp}] {sender}: {content}\n\n")

        # Ensure file was created and return it for download
        if os.path.exists(filename):
            return FileResponse(path=filename, media_type='text/plain', filename=filename)
        else:
            return {"error": "File not created"}
    except Exception as e:
        return {"error": f"Download failed: {str(e)}"}
