#  DL Question Sincerity Analyzer (NLP + Deep Learning + Streamlit)

##  Overview
This project implements a **Deep Learning-based Natural Language Processing (NLP) system** that classifies questions as **Sincere** or **Insincere**.

The system analyzes the **semantic and linguistic patterns** of a question rather than relying on simple keyword matching. It compares multiple modeling approaches, including:

-  Model trained from scratch  
-  Pretrained embeddings (GloVe)  
-  Word2Vec embeddings  

The final system is deployed as an **interactive web application using Streamlit**, allowing real-time predictions.

---

##  Features

-  Classify questions as **Sincere vs Insincere**  
-  Multiple deep learning models (3 approaches)  
-  Compare model performance side-by-side  
-  Confidence score visualization  
-  Interactive web app (Streamlit UI)  
-  Real-time inference  

---

##  How It Works

The system follows a complete NLP pipeline:

### 1. Text Preprocessing
- Convert text to lowercase  
- Remove punctuation using regex  
- Remove stopwords (NLTK)  
- Clean and normalize input text  

> Implemented in the app using a preprocessing function :contentReference[oaicite:0]{index=0}.

---

### 2. Tokenization & Sequencing
- Convert text → numerical sequences  
- Use a trained tokenizer (`tokenizer.pkl`)  
- Pad sequences to fixed length  

---

### 3. Model Architectures

Three different models are used:

-  **Scratch Model** → trained from random embeddings  
-  **GloVe Model** → uses pretrained GloVe embeddings  
-  **Word2Vec Model** → uses Word2Vec embeddings  

All models are loaded dynamically in the app :contentReference[oaicite:1]{index=1}.

---

### 4. Prediction Logic

- Output is a probability score:
  - `> 0.5` →  Insincere  
  - `≤ 0.5` →  Sincere  
- Confidence score is displayed visually  

---

##  Web Application

The project includes a **Streamlit-based interface** that allows users to:

- Enter a question  
- Select one or multiple models  
- Run predictions instantly  
- View results with confidence scores  

> The app uses a clean UI with model selection and real-time analysis :contentReference[oaicite:2]{index=2}.

---

##  Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NLTK**
- **NumPy**
- **Streamlit**
- **Scikit-learn (for preprocessing & embeddings support)**

---

##  Project Structure

```

project/
│── app.py                # Streamlit web app
│── notebook.ipynb        # Model training & experiments
│── tokenizer.pkl         # Tokenizer
│── model_scratch.h5
│── model_glove.h5
│── model_word2vec.h5
│── README.md

````

##  How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/question-classifier.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

---

##  What I Learned

* Building NLP pipelines from scratch
* Working with pretrained embeddings (GloVe & Word2Vec)
* Text preprocessing and tokenization
* Model comparison and evaluation
* Deploying ML models using Streamlit
* Designing user-friendly ML interfaces

---

##  Results

* Successfully classifies questions based on semantic patterns
* Demonstrates differences between embedding techniques
* Provides real-time predictions with confidence scores

---

##  Future Improvements

* Use Transformer-based models (BERT, GPT)
* Improve dataset size and diversity
* Deploy app online (Streamlit Cloud / Hugging Face)
* Add explainability (why prediction was made)
* Improve UI/UX

---

##  Notes

* Model files (`.h5`) and embeddings may not be included due to size constraints
* External resources (e.g., GloVe) must be downloaded separately

---

##  Author

**Laila Tarek**
GitHub: [https://github.com/lailaaa337](https://github.com/lailaaa337)

```

