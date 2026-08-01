import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords nếu chưa có
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()


def lowercase(text):
    return text.lower()


def remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def tokenize(text):
    return text.split()


def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]


def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]


def preprocess_text(text):
    text = lowercase(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    return tokens
