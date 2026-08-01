import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from src.preprocess import preprocess_text
from src.feature import (
    create_dictionary,
    create_word_index,
    create_feature_matrix,
)

DATA_PATH = "data/2cls_spam_text_cls.csv"

MODEL_DIR = "model"

VAL_SIZE = 0.2
TEST_SIZE = 0.125
SEED = 0


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    messages = df["Message"].tolist()
    labels = df["Category"].tolist()

    print("Preprocessing...")

    processed_messages = [
        preprocess_text(message)
        for message in messages
    ]

    dictionary = create_dictionary(processed_messages)

    word_index = create_word_index(dictionary)

    X = create_feature_matrix(
        processed_messages,
        word_index
    )

    encoder = LabelEncoder()

    y = encoder.fit_transform(labels)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_SIZE,
        random_state=SEED,
        shuffle=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True
    )

    print("Training GaussianNB...")

    model = GaussianNB()

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)

    test_pred = model.predict(X_test)

    print(
        "Validation Accuracy:",
        accuracy_score(y_val, val_pred)
    )

    print(
        "Test Accuracy:",
        accuracy_score(y_test, test_pred)
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, "model/model.pkl")
    joblib.dump(dictionary, "model/dictionary.pkl")
    joblib.dump(word_index, "model/word_index.pkl")
    joblib.dump(encoder, "model/encoder.pkl")

    print("Model saved successfully!")


if __name__ == "__main__":
    main()
