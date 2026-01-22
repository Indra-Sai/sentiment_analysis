import streamlit as st
import re
import string
import pickle
import base64
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK resources are available
@st.cache_resource
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                nltk.download(resource)
    
    # Force load wordnet to avoid lazy loading errors
    try:
        from nltk.corpus import wordnet
        wordnet.ensure_loaded()
    except ImportError:
        pass

download_nltk_resources()

def get_base64_of_bin_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load and encode the background image
background_image_path = "background.png"  # Replace with your image path
background_image = get_base64_of_bin_file(background_image_path)

# CSS for the background
page_bg_css = f"""
<style>
    body {{
        background-image: url("data:image/png;base64,{background_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp {{
        background: rgba(0, 0, 0, 0.7); /* Transparency for readability */
        border-radius: 15px;
        padding: 10px;
    }}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_user_input(review):
    tokens = word_tokenize(review.lower())
    # print(tokens)
    
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    # print(tokens)
    
    tokens = [re.sub(r'\b[a-z]\b', "",re.sub(r'[^\w\s | ]', '', word)) for word in tokens]
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # print(tokens)
    
    processed_review = ' '.join(list(filter(lambda x : x , tokens)))
    return processed_review

# Load the trained SVM model and TfidfVectorizer
with open("svm_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Load Roberta pipeline (cached)
@st.cache_resource
def load_roberta():
    from transformers import pipeline
    import torch
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading Roberta on device: {'GPU' if device == 0 else 'CPU'}")
    
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    pipeline_roberta = pipeline("sentiment-analysis", model=model_name, device=device)
    return pipeline_roberta

roberta_pipeline = load_roberta()

# Streamlit App
st.title("Restaurant Review Classifier 🍴")
st.write("Enter a review to classify it as Positive or Negative.")

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_option = st.sidebar.radio("Choose a model:", ("Machine Learning (SVM)", "Transformer (Roberta)"))

# Input review from user
user_review = st.text_area("Your Review", placeholder="Type your restaurant review here...")

# Button to classify the review
if st.button("Classify"):
    if user_review.strip() == "":
        st.warning("Please enter a review!")
    else:
        if model_option == "Machine Learning (SVM)":
            cleaned_review = preprocess_user_input(user_review)
            transformed_review = tfidf.transform([cleaned_review]).toarray()
            prediction = model.predict(transformed_review)[0]

            if prediction == 1:
                st.success("🌟 Positive Review!")
            else:
                st.error("😔 Negative Review.")
        
        else: # Transformer (Roberta)
            # Roberta works best with raw text, no preprocessing needed
            result = roberta_pipeline(user_review)[0]
            label = result['label']
            score = result['score']
            
            # Label mapping for cardiffnlp/twitter-roberta-base-sentiment
            # LABEL_0 -> Negative
            # LABEL_1 -> Neutral
            # LABEL_2 -> Positive
            
            if label == 'LABEL_2':
                st.success(f"🌟 Positive Review! (Confidence: {score:.2f})")
            elif label == 'LABEL_0':
                st.error(f"😔 Negative Review. (Confidence: {score:.2f})")
            else:
                st.info(f"😐 Neutral Review. (Confidence: {score:.2f})")
