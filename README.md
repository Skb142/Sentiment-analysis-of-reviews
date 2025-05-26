# Sentiment-analysis-of-reviews-using-WEB-SCARPING

This project performs sentiment analysis on eBay product reviews using web scraping, natural language processing (NLP), feature engineering with BERT embeddings, and machine learning. It helps in classifying customer feedback into sentiment categories and visualizing insights from user reviews.

---

##  Table of Contents

- [Overview](#overview)
- [Project Phases](#project-phases)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 📖 Overview

This project is divided into 4 main phases:

1. **Web Scraping** eBay reviews using Selenium.
2. **Data Cleaning + Sentiment Analysis + Visualization** using TextBlob and Matplotlib/Seaborn.
3. **Feature Engineering** using BERT embeddings and handcrafted features.
4. **Model Training** using machine learning models like Logistic Regression, Random Forest, and SVM.

---

## 🚀 Project Phases

### ✅ Phase 1: Web Scraping

- Uses Selenium to scrape:
  - Review title
  - Review content
  - Ratings
- Stores data into `ebay_reviews.csv`

### ✅ Phase 2: Sentiment Analysis + EDA

- Cleans review text
- Applies TextBlob to extract sentiment polarity
- Visualizes:
  - Word clouds
  - Review length distributions
  - Word counts

### ✅ Phase 3: Feature Engineering

- BERT embeddings from `all-MiniLM-L6-v2`
- Sentiment polarity score
- Negation count (to detect sarcasm)

### ✅ Phase 4: Model Training & Evaluation

- Models used:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
- Evaluation:
  - Accuracy, Precision, Recall, F1-score, ROC-AUC
- Best model saved as `classifier.pkl`

---

## 🛠️ Technologies Used

- **Language:** Python
- **Scraping:** Selenium, BeautifulSoup
- **NLP:** NLTK, TextBlob, Sentence-Transformers
- **ML Models:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn, WordCloud

---
# 🛒 eBay Sentiment Analysis: Scraping + NLP + Machine Learning

This project performs sentiment analysis on eBay product reviews using web scraping, natural language processing (NLP), feature engineering with BERT embeddings, and machine learning. It helps in classifying customer feedback into sentiment categories and visualizing insights from user reviews.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Phases](#project-phases)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 📖 Overview

This project is divided into 4 main phases:

1. **Web Scraping** eBay reviews using Selenium.
2. **Data Cleaning + Sentiment Analysis + Visualization** using TextBlob and Matplotlib/Seaborn.
3. **Feature Engineering** using BERT embeddings and handcrafted features.
4. **Model Training** using machine learning models like Logistic Regression, Random Forest, and SVM.

---

## 🚀 Project Phases

### ✅ Phase 1: Web Scraping

- Uses Selenium to scrape:
  - Review title
  - Review content
  - Ratings
- Stores data into `ebay_reviews.csv`

### ✅ Phase 2: Sentiment Analysis + EDA

- Cleans review text
- Applies TextBlob to extract sentiment polarity
- Visualizes:
  - Word clouds
  - Review length distributions
  - Word counts

### ✅ Phase 3: Feature Engineering

- BERT embeddings from `all-MiniLM-L6-v2`
- Sentiment polarity score
- Negation count (to detect sarcasm)

### ✅ Phase 4: Model Training & Evaluation

- Models used:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
- Evaluation:
  - Accuracy, Precision, Recall, F1-score, ROC-AUC
- Best model saved as `classifier.pkl`

---

## 🛠️ Technologies Used

- **Language:** Python
- **Scraping:** Selenium, BeautifulSoup
- **NLP:** NLTK, TextBlob, Sentence-Transformers
- **ML Models:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn, WordCloud

---

"This is my copy of the project" 
