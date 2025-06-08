# 🧠 Socratic Chat

**Socratic Chat** is a web-based AI application that simulates Socratic-style dialogue using OpenAI's language models and an ML-based input classifier. The app encourages critical thinking by prompting users with philosophical questions in response to their input.

---

## 🚀 Features

- 🔁 Socratic interaction powered by GPT (via OpenAI API)
- 🧠 ML-based input categorization (Decision Tree Classifier)
- ✍️ Preprocessing with spaCy (lemmatization, stopword removal)
- 📅 Timestamped messages
- ♻️ Reset conversation
- 📄 Download full chat history (.txt)
- 🌐 Minimal responsive web UI with Bootstrap

---

## 📁 Project Structure

| File               | Purpose                                              |
| ------------------ | ---------------------------------------------------- |
| `main.py`          | FastAPI app — API routes, chat logic, HTML rendering |
| `requirements.txt` | All dependencies needed to run the app               |
| `model.pkl`        | Trained scikit-learn classifier (e.g., DecisionTree) |
| `vectorizer.pkl`   | Fitted TF-IDF vectorizer for input text              |
| `nltk_download.py` | Downloads required NLTK resources (optional)         |

---

## 🛠️ Installation

### 📦 Requirements

Install dependencies via `requirements.txt`:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> 💡 Make sure to set the environment variables `OPENAI_API_KEY` and `OPENAI_PROJECT_ID`.

---

## 💻 Run Locally

```bash
# Clone the repo
git clone https://github.com/yasamin0/socratic-ai
cd socratic-ai

# Optional: create and activate virtual environment
python -m venv venv
source venv/bin/activate    # on macOS/Linux
# OR
venv\Scripts\activate       # on Windows

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run the app
uvicorn main:app --reload
```

Visit: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🧠 How It Works

1. User submits a question via form.
2. Input is lemmatized and classified by a trained Decision Tree (e.g., "factual", "emotional", etc.).
3. The GPT model generates a response guided by a Socratic system prompt.
4. Responses and questions are timestamped and stored in memory.

---

## 📤 Deployment

You can deploy this app using services like:

- **Render** (recommended)
- **Vercel** (with serverless adaptation)
- **Streamlit Cloud** (if redesigned as a Streamlit app)
- **Heroku** (for small deployments)

---

## 📑 Environment Variables

Ensure the following are set (e.g. in `.env` or host panel):

```env
OPENAI_API_KEY=your-key-here
OPENAI_PROJECT_ID=your-project-id-here
```

---

## 📥 Download & Reset

- **Reset Chat**: Clears the in-memory conversation state.
- **Download Chat**: Downloads a `.txt` file with timestamped chat history.

---

## ✅ Evaluation Criteria Coverage

- ✅ Python + FastAPI
- ✅ Integrated GPT via OpenAI API
- ✅ NLP preprocessing (spaCy)
- ✅ ML classification
- ✅ Error handling
- ✅ Clean documentation (this README)
- ✅ Simple Bootstrap frontend
- ✅ Ready for deployment

---

## 📘 License

MIT License — free for personal and academic use.

---

## 👤 Author

Developed with curiosity by **Yasamin H. Sani**
