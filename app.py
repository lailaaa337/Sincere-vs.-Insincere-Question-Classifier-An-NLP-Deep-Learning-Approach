import streamlit as st
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
# Import Keras components from tensorflow to ensure compatibility
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---

# Set wide layout for cleaner use of space
st.set_page_config(
    page_title="DL Question Classifier",
    page_icon="🤖",
    layout="wide"
)

# Constants
MAX_LEN = 50 
MODEL_PATHS = {
    "Scratch Model": "model_scratch.h5",
    "GloVe Model": "model_glove.h5",
    "Word2Vec Model": "model_word2vec.h5"
}

# --- RESOURCE LOADING ---

@st.cache_resource
def load_resources():
    """Load tokenizer, all three models, and NLTK resources."""
    try:
        # 1. NLTK Download and STOPWORDS Initialization (FIXED LOCATION)
        nltk.download('stopwords', quiet=True)
        STOPWORDS = set(stopwords.words('english')) # <--- MOVED HERE

        # 2. Load Tokenizer
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        
        # 3. Load Models
        models = {}
        for name, path in MODEL_PATHS.items():
            models[name] = load_model(path)
        
        # Return all loaded resources
        return tokenizer, models, STOPWORDS
    
    except Exception as e:
        st.error(f"Error loading required files: {str(e)}")
        st.error("Please verify that tokenizer.pkl and all .h5 model files exist in the same directory.")
        return None, None, None

def clean_text(text, STOPWORDS):
    """Clean and preprocess text (now requires STOPWORDS as argument)."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = [word for word in text.split() if word not in STOPWORDS]
    return ' '.join(words)

# --- MAIN APP LOGIC ---

def main():
    
    # Updated call to load_resources
    tokenizer, models, STOPWORDS = load_resources()
    
    # 1. TITLE AND HEADER (Main Area)
    st.title("❓ DL Question Sincerity Analyzer")
    st.markdown("---")
    
    if tokenizer is None:
        return 

    # 2. USER INPUT AND CONTROLS (Single Column, Simple)
    
    with st.container(border=True):
        st.subheader("1. Enter Question")
        question = st.text_area(
            "Type your question here:",
            height=100,
            placeholder="e.g. 'What is the most effective way to learn Python programming?'",
            key="question_input",
            label_visibility="collapsed"
        )

        st.subheader("2. Select Model(s) for Prediction")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            model_choice = st.selectbox(
                "Choose Model Set:",
                ["All Models", "Scratch Model", "GloVe Model", "Word2Vec Model"],
                index=0,
                label_visibility="collapsed"
            )

        with col2:
            analyze_btn = st.button(
                "🚀 Run Analysis",
                type="primary",
                use_container_width=True
            )
    
    # 3. PREDICTION LOGIC AND RESULTS
    
    if analyze_btn and question:
        
        # Preprocessing (Pass STOPWORDS to clean_text)
        cleaned = clean_text(question, STOPWORDS) # <--- UPDATED CALL
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # Determine models to run
        if model_choice == "All Models":
            models_to_run = models.items()
        else:
            models_to_run = [(model_choice, models[model_choice])]
        
        st.subheader("📊 Prediction Results")
        
        # Display results in columns
        cols = st.columns(len(models_to_run))
        
        for idx, (name, model) in enumerate(models_to_run):
            try:
                # Prediction
                pred = model.predict(padded, verbose=0)[0][0]
                
                # Sincere = 0 (low prediction), Insincere = 1 (high prediction)
                if pred > 0.5:
                    result_text = "❌ INSINCERE"
                    confidence = pred
                else:
                    result_text = "✅ SINCERE"
                    confidence = 1 - pred
                
                with cols[idx]:
                    if result_text.startswith("✅"):
                         st.success(f"**{name}:** {result_text}")
                    else:
                         st.error(f"**{name}:** {result_text}")

                    st.metric("Confidence Score", f"{confidence:.2%}")
                    st.progress(float(confidence))
                        
            except Exception as e:
                with cols[idx]:
                    st.exception(f"Error predicting with {name}")
    
    elif analyze_btn and not question:
        st.warning("⚠️ Please enter a question to analyze.")
        
    st.markdown("---")
    st.info("💡 **Project Note**: This page demonstrates the inference step for all trained models, fulfilling the Web App deployment criterion.")

if __name__ == "__main__":
    main()