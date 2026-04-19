# Sincere vs Insincere Question Classifier

## Overview

This project is about building a model that can classify questions as sincere or insincere using NLP and deep learning.

I worked on different approaches including training a model from scratch and also using pretrained embeddings like GloVe and Word2Vec. The goal was to compare how each method performs on this task.

---

## What I did

* Preprocessed text (cleaning, removing stopwords, etc.)
* Converted text into sequences using a tokenizer
* Built multiple deep learning models
* Used GloVe and Word2Vec embeddings
* Evaluated predictions based on confidence scores

---

## Models used

* Scratch model
* GloVe-based model
* Word2Vec-based model

---

## Web App

I created a simple Streamlit app where you can:

* Enter any question
* Choose which model to use
* See if the question is sincere or not

---

## How to run

Clone the repo:

```bash id="7o9db6"
git clone https://github.com/lailaaa337/Sincere-vs.-Insincere-Question-Classifier-An-NLP-Deep-Learning-Approach.git
```

Install requirements:

```bash id="6wbvnj"
pip install -r requirements.txt
```

Run the app:

```bash id="j6c8pw"
streamlit run app.py
```

---

## Notes

The dataset, embeddings, and model files are not included because they are too large for GitHub.

You can download GloVe from:
https://nlp.stanford.edu/projects/glove/

---

## Future improvements

* Try more advanced models (like transformers)
* Improve accuracy
* Deploy the app online

---

## Author

Laila Tarek
