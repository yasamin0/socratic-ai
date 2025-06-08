from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import pickle

data = [
    ("What is the meaning of life?", "philosophy"),
    ("Do we have free will?", "philosophy"),
    ("Should we always tell the truth?", "ethics"),
    ("Is stealing ever justified?", "ethics"),
    ("How do we know what we know?", "epistemology"),
    ("What makes an argument valid?", "logic"),
    ("How should we treat others?", "ethics"),
    ("What is love?", "life"),
    ("What is consciousness?", "philosophy"),
    ("What is friendship?", "life")
]

texts, labels = zip(*data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = DecisionTreeClassifier()
model.fit(X, labels)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
