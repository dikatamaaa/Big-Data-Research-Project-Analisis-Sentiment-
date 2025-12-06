
# Political Sentiment Analysis: Indonesian Presidential Candidates 2019

## 📊 Overview
This project implements sentiment analysis on tweets about Indonesian presidential candidates from the 2019 election using two machine learning approaches: Support Vector Machines (SVM) and Bidirectional Long Short-Term Memory (BI-LSTM). The goal is to compare traditional machine learning with deep learning techniques for Indonesian-language social media sentiment classification.

## 🎯 Objectives
* Compare performance between SVM and BI-LSTM for Indonesian sentiment analysis
* Analyze public sentiment toward presidential candidates from Twitter data
* Implement comprehensive text preprocessing pipeline for Indonesian language
* Visualize sentiment distribution and key topics

## 📁 Dataset
* Source: Indonesian Presidential Election 2019 Tweet Dataset
* Size: 1,815 tweets (balanced to 596 negative + 596 positive after undersampling)
* Features:
  * Text Tweet: Raw tweet text in Indonesian
  * Sentiment: Binary label (0 = Negative, 1 = Positive)
* Preprocessing: Data cleaning, duplicate removal, undersampling for class balance

## 🛠️ Tech Stack
* Programming: Python 3.8+
* ML Libraries: Scikit-learn, TensorFlow/Keras, Pandas, NumPy
* NLP Tools: NLTK, Gensim (Word2Vec), Sastrawi (Indonesian stemmer)
* Visualization: Matplotlib, Seaborn, WordCloud
* Vectorization: CountVectorizer, Word2Vec Embeddings

## 📈 Methodology

    1. Data Preprocessing Pipeline
    Raw Tweets → Remove URLs/numbers/special chars → Lowercasing → Stopword removal → Tokenization → Stemming → Cleaned text

    2. Feature Engineering
    For SVM: CountVectorizer (Bag-of-Words representation)
    For BI-LSTM: Word2Vec embeddings (300 dimensions, 43,981 vocabulary)

    3. Model Architectures
    SVM: RBF kernel, gamma='auto'
    BI-LSTM:
      - Bidirectional LSTM layers
      - Embedding layer (Word2Vec)
      - Dense layers with dropout
      - Adam optimizer, batch_size=64, epochs=100

    4. Evaluation Metrics
      - Accuracy, Precision, Recall, F1-Score
      - Confusion Matrix
      - ROC-AUC Score

## 📊 Results
Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
| ---------- | ---------- | ---------- | ---------- | ---------- |
| BI-LSTM | 71.4% | 0.72 | 0.71 | 0.71 |
| SVM | 59.1% | 0.60 | 0.59% | 0.59 |

Key Findings
* BI-LSTM outperforms SVM by ~12% accuracy for Indonesian tweet sentiment analysis
* Deep learning captures contextual nuances better in Indonesian language
* Sequence-aware models (LSTM) handle Indonesian sentence structures more effectively
* Traditional ML (SVM) struggles with informal, unstructured social media text

## 📊 Visualization
Word Clouds
* Positive Sentiment: "presiden", "baik", "bangsa", "maju", "kerja"
* Negative Sentiment: "korupsi", "masalah", "janji", "rakyat", "negara"
