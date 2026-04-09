# Disaster Tweet Classification using Classical NLP & Word Embeddings
This project is an end-to-end NLP system that classifies tweets as disaster-related or not using multiple text representation techniques and machine learning models. 

The system compares traditional methods like Bag-of-Words and TF-IDF with semantic embeddings such as GloVe and TF-IDF weighted GloVe, highlighting the strengths and limitations of each approach.

An interactive Streamlit web application is built to allow real-time predictions, along with confidence scores and uncertainty detection for better interpretability.

## 📊 Model Performance

| Model              | Accuracy | F1 Score |
|-------------------|----------|----------|
| Bag-of-Words      | 0.81     | 0.77     |
| TF-IDF            | 0.80     | 0.76     |
| GloVe             | 0.79     | 0.75     |
| TF-IDF + GloVe    | 0.77     | 0.73     |

## 🧠 Key Insights

- Bag-of-Words outperformed more complex models due to strong keyword signals in disaster tweets  
- TF-IDF slightly underperformed as it downweights frequent but important words  
- GloVe embeddings captured semantic meaning but lost information through averaging  
- TF-IDF weighted GloVe did not outperform simpler models due to alignment and weighting limitations  
- Models struggle with sarcasm and context-dependent expressions

## 🖥️ Demo

The Streamlit app allows users to:

- Enter a tweet  
- Select a model  
- View prediction with confidence score  

Example:
- Input: "earthquake again just perfect timing"  
- Output: Not Disaster (Confidence: 0.52)  
- Insight: Model is uncertain

## ⚙️ Tech Stack

- Python  
- Scikit-learn  
- NLTK  
- NumPy  
- Streamlit

## 🚀 How to Run

1. Clone the repository  
2. Install dependencies:

   pip install -r requirements.txt

3. Run the app:

   streamlit run app.py

## 🚀 Future Improvements

- Use transformer models (BERT) for better context understanding  
- Improve sarcasm detection  
- Add explainability (feature importance visualization)  
- Deploy with scalable backend  
