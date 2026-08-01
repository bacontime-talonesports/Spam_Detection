import numpy as np


def create_dictionary(messages):
    """
    Tạo dictionary từ toàn bộ tập dữ liệu.
    """
    dictionary = sorted(set(token for tokens in messages for token in tokens))
    return dictionary


def create_word_index(dictionary):
    """
    Tạo bảng tra cứu từ -> chỉ số.
    """
    return {word: idx for idx, word in enumerate(dictionary)}


def create_features(tokens, word_index):
    """
    Chuyển một email thành vector Bag of Words.
    """
    features = np.zeros(len(word_index), dtype=np.float32)

    for token in tokens:
        if token in word_index:
            features[word_index[token]] += 1

    return features


def create_feature_matrix(messages, word_index):
    """
    Chuyển toàn bộ dataset thành ma trận đặc trưng.
    """
    X = np.array(
        [create_features(tokens, word_index) for tokens in messages],
        dtype=np.float32,
    )

    return X
