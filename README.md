# Socratic Chat

**Socratic Chat** is a web-based chatbot powered by OpenAI's GPT-4 and a machine learning classifier that simulates Socratic-style dialogue. It analyzes the user's input, categorizes it, and responds with thoughtful philosophical prompts.

---

## ✅ Features

- 💬 Chat interface styled with Bootstrap
- 🧠 GPT-4o-mini-based Socratic responses
- 🗃️ ML-based question categorization (e.g., philosophical, emotional, factual)
- 🕒 Message timestamps
- ♻️ Reset chat history
- 📥 Download conversation log

---

## 📦 Requirements

Install dependencies:

```bash
pip install fastapi uvicorn openai spacy scikit-learn
python -m spacy download en_core_web_sm
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/yasamin0/socratic-ai
cd socratic-ai

# (Optional) Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate for macOS/Linux

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start the server
uvicorn main:app --reload
```

Visit: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📁 Project Structure

| File               | Description                           |
| ------------------ | ------------------------------------- |
| `main.py`          | FastAPI backend & frontend logic      |
| `ml_model.py`      | Script to train and save the ML model |
| `model.pkl`        | Pickled trained classifier            |
| `vectorizer.pkl`   | Pickled TF-IDF vectorizer             |
| `nltk_download.py` | Downloads necessary NLTK resources    |

---

## 🔑 Notes

- Insert your OpenAI API key in `main.py`
- Chat history is session-based (in-memory)
- ML categorization uses a simple trained `DecisionTreeClassifier`

---

## 📃 License

MIT — Free for academic/personal use.

---

## 🙋 Author

Built with curiosity by Yasamin H. Sani
