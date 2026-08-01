import joblib

from src.preprocess import preprocess_text
from src.feature import create_features


class SpamDetector:

    def __init__(self):

        self.model = joblib.load("model/model.pkl")

        self.word_index = joblib.load(
            "model/word_index.pkl"
        )

        self.encoder = joblib.load(
            "model/encoder.pkl"
        )

    def predict(self, text):

        tokens = preprocess_text(text)

        x = create_features(
            tokens,
            self.word_index
        )

        x = x.reshape(1, -1)

        prediction = self.model.predict(x)[0]

        label = self.encoder.inverse_transform(
            [prediction]
        )[0]

        return label

    def predict_probability(self, text):

        tokens = preprocess_text(text)

        x = create_features(
            tokens,
            self.word_index
        )

        x = x.reshape(1, -1)

        probs = self.model.predict_proba(x)[0]

        prediction = probs.argmax()

        confidence = probs[prediction]

        label = self.encoder.inverse_transform(
            [prediction]
        )[0]

        return label, confidence
