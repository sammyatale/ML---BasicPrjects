import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# -----------------------------
# 1. Load Shakespeare dataset
# -----------------------------

filepath = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)

text = open(filepath, "rb").read().decode(encoding="utf-8").lower()

# Use first 1 million characters
text = text[:1_000_000]

print(f"Characters in dataset: {len(text)}")


# -----------------------------
# 2. Create character mappings
# -----------------------------

characters = sorted(set(text))

char_to_index = {char: index for index, char in enumerate(characters)}
index_to_char = {index: char for index, char in enumerate(characters)}

print(f"Unique characters: {len(characters)}")


# -----------------------------
# 3. Create training sequences
# -----------------------------

SEQ_LENGTH = 40
STEP_SIZE = 3

sentences = []
next_chars = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i:i + SEQ_LENGTH])
    next_chars.append(text[i + SEQ_LENGTH])

print(f"Training sequences: {len(sentences)}")


# -----------------------------
# 4. One-hot encode data
# -----------------------------

x = np.zeros(
    (len(sentences), SEQ_LENGTH, len(characters)),
    dtype=bool
)

y = np.zeros(
    (len(sentences), len(characters)),
    dtype=bool
)

for i, sentence in enumerate(sentences):

    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1

    y[i, char_to_index[next_chars[i]]] = 1


print(f"X shape: {x.shape}")
print(f"Y shape: {y.shape}")


# -----------------------------
# 5. Build LSTM model
# -----------------------------

model = Sequential()

model.add(
    LSTM(
        256,
        input_shape=(SEQ_LENGTH, len(characters))
    )
)

model.add(Dense(len(characters)))

model.add(Activation("softmax"))


# -----------------------------
# 6. Compile model
# -----------------------------

model.compile(
    loss="categorical_crossentropy",
    optimizer=RMSprop(learning_rate=0.01)
)
start_time = time.time()


# -----------------------------
# 7. Train model
# -----------------------------

# -----------------------------
# 7. Train model
# -----------------------------

start_time = time.time()

history = model.fit(
    x,
    y,
    batch_size=64,
    epochs=8
)

training_time = time.time() - start_time


# -----------------------------
# 8. Save training history
# -----------------------------

with open(OUTPUTS_DIR / "training_history_v2.json", "w") as file:
    json.dump(history.history, file)


# -----------------------------
# 9. Save model
# -----------------------------

model.save(MODELS_DIR / "shakespeare_lstm_v2.keras")


# -----------------------------
# 10. Save experiment details
# -----------------------------

experiment = {
    "version": "v2",
    "lstm_units": 256,
    "sequence_length": SEQ_LENGTH,
    "step_size": STEP_SIZE,
    "batch_size": 64,
    "epochs": 8,
    "learning_rate": 0.01,
    "training_time_seconds": training_time,
    "final_loss": history.history["loss"][-1],
    "model_path": "models/shakespeare_lstm_v2.keras"
}

with open(OUTPUTS_DIR / "experiment_v2.json", "w") as file:
    json.dump(experiment, file, indent=4)

print("Model saved successfully!")
print(f"Training time: {training_time:.2f} seconds")
print(f"Final loss: {history.history['loss'][-1]:.4f}")
