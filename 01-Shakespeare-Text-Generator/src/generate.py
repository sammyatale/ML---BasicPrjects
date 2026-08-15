import random
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Load Shakespeare dataset
filepath = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)

text = open(filepath, "rb").read().decode("utf-8").lower()

text = text[:1_000_000]


# Character mappings
characters = sorted(set(text))

char_to_index = {
    char: index
    for index, char in enumerate(characters)
}

index_to_char = {
    index: char
    for index, char in enumerate(characters)
}


# Load trained model
model = tf.keras.models.load_model(
    PROJECT_ROOT / "models" / "shakespeare_lstm_v2.keras"
)

SEQ_LENGTH = 40

def sample(preds, temperature=1.0):

    preds = np.asarray(preds).astype("float64")

    preds = np.log(preds + 1e-7) / temperature

    exp_preds = np.exp(preds)

    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)


def generate_text(length=400, temperature=0.5):

    start_index = random.randint(
        0,
        len(text) - SEQ_LENGTH - 1
    )

    sentence = text[
        start_index:start_index + SEQ_LENGTH
    ]

    generated = sentence

    for _ in range(length):

        X = np.zeros(
            (1, SEQ_LENGTH, len(characters))
        )

        for t, char in enumerate(sentence):
            X[0, t, char_to_index[char]] = 1

        preds = model.predict(
            X,
            verbose=0
        )[0]

        next_index = sample(
            preds,
            temperature
        )

        next_character = index_to_char[next_index]

        generated += next_character

        sentence = sentence[1:] + next_character

    return generated

print("========== SHAKESPEARE TEXT GENERATOR ==========")

temperatures = [0.2, 0.5, 1.0, 2.0]

for temperature in temperatures:

    print(f"\n========== Temperature: {temperature} ==========\n")

    print(
        generate_text(
            length=400,
            temperature=temperature
        )
    )
