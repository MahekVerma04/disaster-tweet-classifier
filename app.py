import streamlit as st
import pickle
import numpy as np
import re
import string
from nltk.corpus import stopwords



# ---------------- LOAD ---------------- #
tfidf = pickle.load(open("tfidf.pkl", "rb"))
bow = pickle.load(open("bow.pkl", "rb"))

tfidf_model = pickle.load(open("tfidf_model.pkl", "rb"))
bow_model = pickle.load(open("bow_model.pkl", "rb"))
glove_model = pickle.load(open("glove_model.pkl", "rb"))
weighted_glove_model = pickle.load(open("weighted_glove_model.pkl", "rb"))

embeddings = pickle.load(open("glove_embeddings.pkl", "rb"))

@st.cache_resource
def load_embeddings():
    return pickle.load(open("glove_reduced.pkl", "rb"))

embeddings = load_embeddings()

stop_words = set(stopwords.words("english"))


def get_confidence(model, vec):
    # For Logistic Regression
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        return max(proba)
    
    # For LinearSVC
    elif hasattr(model, "decision_function"):
        score = model.decision_function(vec)[0]
        return 1 / (1 + np.exp(-score))  # sigmoid
    
    return None


chat_words = {'a3': 'anytime, anywhere, anyplace',
 'adih': 'another day in hell',
 'afk': 'away from keyboard',
 'afaik': 'as far as i know',
 'asap': 'as soon as possible',
 'asl': 'age, sex, location',
 'atk': 'at the keyboard',
 'atm': 'at the moment',
 'bae': 'before anyone else',
 'bak': 'back at keyboard',
 'bbl': 'be back later',
 'bbs': 'be back soon',
 'bfn': 'bye for now',
 'b4n': 'bye for now',
 'brb': 'be right back',
 'bruh': 'bro',
 'brt': 'be right there',
 'bsaaw': 'big smile and a wink',
 'btw': 'by the way',
 'bwl': 'bursting with laughter',
 'csl': 'can’t stop laughing',
 'cu': 'see you',
 'cul8r': 'see you later',
 'cya': 'see you',
 'dm': 'direct message',
 'faq': 'frequently asked questions',
 'fc': 'fingers crossed',
 'fimh': 'forever in my heart',
 'fomo': 'fear of missing out',
 'fr': 'for real',
 'fwiw': "for what it's worth",
 'fyp': 'for you page',
 'fyi': 'for your information',
 'g9': 'genius',
 'gal': 'get a life',
 'gg': 'good game',
 'gmta': 'great minds think alike',
 'gn': 'good night',
 'goat': 'greatest of all time',
 'gr8': 'great!',
 'hbd': 'happy birthday',
 'ic': 'i see',
 'icq': 'i seek you',
 'idc': 'i don’t care',
 'idk': "i don't know",
 'ifyp': 'i feel your pain',
 'ilu': 'i love you',
 'ily': 'i love you',
 'imho': 'in my honest/humble opinion',
 'imu': 'i miss you',
 'imo': 'in my opinion',
 'iow': 'in other words',
 'irl': 'in real life',
 'iykyk': 'if you know, you know',
 'jk': 'just kidding',
 'kiss': 'keep it simple, stupid',
 'l': 'loss',
 'l8r': 'later',
 'ldr': 'long distance relationship',
 'lmk': 'let me know',
 'lmao': 'laughing my a** off',
 'lol': 'laughing out loud',
 'ltns': 'long time no see',
 'm8': 'mate',
 'mfw': 'my face when',
 'mid': 'mediocre',
 'mrw': 'my reaction when',
 'mte': 'my thoughts exactly',
 'nvm': 'never mind',
 'nrn': 'no reply necessary',
 'npc': 'non-player character',
 'oic': 'oh i see',
 'op': 'overpowered',
 'pita': 'pain in the a**',
 'pov': 'point of view',
 'prt': 'party',
 'prw': 'parents are watching',
 'rofl': 'rolling on the floor laughing',
 'roflol': 'rolling on the floor laughing out loud',
 'rotflmao': 'rolling on the floor laughing my a** off',
 'rn': 'right now',
 'sk8': 'skate',
 'stats': 'your sex and age',
 'sus': 'suspicious',
 'tbh': 'to be honest',
 'tfw': 'that feeling when',
 'thx': 'thank you',
 'time': 'tears in my eyes',
 'tldr': 'too long, didn’t read',
 'tntl': 'trying not to laugh',
 'ttfn': 'ta-ta for now!',
 'ttyl': 'talk to you later',
 'u': 'you',
 'u2': 'you too',
 'u4e': 'yours for ever',
 'w': 'win',
 'w8': 'wait...',
 'wb': 'welcome back',
 'wtf': 'what the f**k',
 'wtg': 'way to go!',
 'wuf': 'where are you from?',
 'wyd': 'what you doing?',
 'wywh': 'wish you were here',
 'zzz': 'sleeping, bored, tired'}

import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))



def clean_text(text):
    
    # 1. lowercase
    text = text.lower()
    
    # 2. remove html
    text = re.sub(r'<.*?>', '', text)
    
    # 3. remove urls
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # 4. replace chat words
    words = text.split()
    words = [chat_words[word] if word in chat_words else word for word in words]
    text = " ".join(words)
    
    # 5. remove punctuation
    #text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 6. tokenize
    words = text.split()
    
    # 7. remove stopwords and stem
    #words = [ps.stem(word) for word in words if word not in stop_words]
    
    # 8. join back
    text = " ".join(words)
    
    return text


def clean_text_glove(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    
    words = text.split()
    words = [chat_words[word] if word in chat_words else word for word in words]
    
    text = " ".join(words)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return words   # 👈 IMPORTANT (list, not string)

def get_glove_vector(tokens):

    vectors = []

    for word in tokens:

        if word in embeddings:
            vectors.append(embeddings[word])

    if len(vectors) == 0:
        return np.zeros(300)

    return np.mean(vectors, axis=0)


def get_weighted_glove_vector(tokens, text):

    vector = np.zeros(300)
    weight_sum = 0

    #tfidf_vector = tfidf.transform([text]).toarray()[0]
    tfidf_vector = tfidf.transform([text])

    for word in tokens:
        if word in embeddings and word in tfidf.vocabulary_:
            idx = tfidf.vocabulary_[word]
            weight = tfidf_vector[0,idx]

            if weight > 0:
                vector += embeddings[word] * weight
                weight_sum += weight

    if weight_sum != 0:
        vector /= weight_sum

    return vector

# ---------------- UI ---------------- #
st.title("🚨 Disaster Tweet Classifier")

tweet = st.text_area("Enter a tweet")

model_choice = st.selectbox(
    "Choose model",
    ["BoW", "TF-IDF", "GloVe", "Weighted GloVe"]
)



if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet")
    else:
        # -------- MODEL SELECTION -------- #
        if model_choice == "BoW":
            cleaned = clean_text(tweet)
            vec = bow.transform([cleaned])
            pred = bow_model.predict(vec)[0]
            confidence = get_confidence(bow_model, vec)

        elif model_choice == "TF-IDF":
            cleaned = clean_text(tweet)
            vec = tfidf.transform([cleaned])
            pred = tfidf_model.predict(vec)[0]
            confidence = get_confidence(tfidf_model, vec)

        elif model_choice == "GloVe":
            tokens = clean_text_glove(tweet)
            vec = get_glove_vector(tokens)
            pred = glove_model.predict([vec])[0]
            confidence = get_confidence(glove_model, [vec])

        else:  # Weighted GloVe
            tokens = clean_text_glove(tweet)
            text = " ".join(tokens)
            vec = get_weighted_glove_vector(tokens, text)
            pred = weighted_glove_model.predict([vec])[0]
            confidence = get_confidence(weighted_glove_model, [vec])

        # -------- OUTPUT -------- #
        if pred == 1:
            st.error(f"🚨 Disaster (Confidence: {confidence:.2f})")
        else:
            st.success(f"🙂 Not a Disaster (Confidence: {confidence:.2f})")

        # -------- CONFIDENCE BAR -------- #
        st.progress(float(confidence))

        # -------- UNCERTAINTY WARNING -------- #
        if 0.4 < confidence < 0.6:
            st.warning("⚠️ Model is unsure about this prediction")
import os
print(os.getcwd())