import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import re
import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure nltk resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load Data
print("Loading data...")
df = pd.read_csv('Restaurant_Reviews.tsv', sep='\t')

# --- SVM Section ---
print("Loading SVM model...")
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_user_input(review):
    tokens = word_tokenize(review.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    tokens = [re.sub(r'\b[a-z]\b', "",re.sub(r'[^\w\s | ]', '', word)) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    processed_review = ' '.join(list(filter(lambda x : x , tokens)))
    return processed_review

print("Predicting with SVM...")
y_true = df['Liked'].values
y_pred_svm = []

for text in df['Review']:
    cleaned = preprocess_user_input(text)
    transformed = vectorizer.transform([cleaned]).toarray()
    pred = svm_model.predict(transformed)[0]
    y_pred_svm.append(pred)

# --- Roberta Section ---
print("Loading Roberta model...")
# Using a specific model for binary sentiment if possible, or keeping the user request for "Roberta"
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
# This model outputs LABEL_0 (Negative), LABEL_1 (Neutral), LABEL_2 (Positive)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

print("Predicting with Roberta...")
y_pred_roberta = []

for text in df['Review']:
    # Truncate to 512 characters roughly to avoid token length issues if any, though pipeline handles truncation
    res = sentiment_pipeline(text, truncation=True, max_length=512)[0]
    label = res['label']
    
    # Map labels
    if label == 'LABEL_0': # Negative
        y_pred_roberta.append(0)
    elif label == 'LABEL_2': # Positive
        y_pred_roberta.append(1)
    else:
        # Neutral - This is tricky for binary classification.
        # Mapping to 0 as "not liked" (or we could ignore these rows, but we need to match y_true length)
        # For 'Liked', usually 1 is positive, 0 is negative. Neutral is closer to 0 (didn't like it enough to say yes).
        y_pred_roberta.append(0) 

# --- Confusion Matrix ---
print("Generating confusion matrices...")

def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

plot_cm(y_true, y_pred_svm, 'Confusion Matrix - SVM', 'confusion_matrix_svm.png')
plot_cm(y_true, y_pred_roberta, 'Confusion Matrix - Roberta', 'confusion_matrix_roberta.png')

print("Done. Images saved.")
