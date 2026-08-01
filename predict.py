import joblib
import numpy as np

from src.preprocess import preprocess_text
from src.feature import create_features


class SpamDetector:

    def __init__(self):

        self.model = joblib.load("model/model.pkl")
        self.dictionary = joblib.load("model/dictionary.pkl")
        self.encoder = joblib.load("model/encoder.pkl")


    def predict(self, text):

        tokens = preprocess_text(text)

        features = create_features(tokens, self.dictionary)

        features = np.array(features).reshape(1, -1)

        prediction = self.model.predict(features)

        label = self.encoder.inverse_transform(prediction)[0]

        return label


    def predict_probability(self, text):

        tokens = preprocess_text(text)

        features = create_features(tokens, self.dictionary)

        features = np.array(features).reshape(1, -1)

        prediction = self.model.predict(features)

        probability = self.model.predict_proba(features)

        label = self.encoder.inverse_transform(prediction)[0]

        confidence = np.max(probability)

        return label, confidence
