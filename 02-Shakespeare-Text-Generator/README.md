# Shakespeare Character-Level Text Generator

A character-level LSTM that generates Shakespeare-style text one character at a time. It is implemented with TensorFlow/Keras and trained on the public-domain Shakespeare corpus supplied by TensorFlow.

## How it works

The training script lowercases the first 1,000,000 characters of the corpus, creates overlapping 40-character input sequences (stride 3), and one-hot encodes both the sequences and their next characters. An LSTM learns to predict the next character through a softmax output layer. During generation, each sampled character is appended to the context window and used to predict the next one.

## Dataset

The scripts download `shakespeare.txt` from TensorFlow's hosted dataset URL on first use. The dataset is not stored in this repository.

## Architecture and training

- Sequence length: 40 characters
- Step size: 3 characters
- Optimizer: RMSprop (`learning_rate=0.01`)
- Output: softmax distribution over the corpus character vocabulary

| Version | LSTM units | Epochs | Batch size | Final training loss | Model |
|---|---:|---:|---:|---:|---|
| V1 | 128 | 4 | 26 | ~1.6590 | `models/shakespearetextgenerator.h5` |
| V2 | 256 | 8 | 64 | ~1.4293 | `models/shakespeare_lstm_v2.keras` |

V2 has 626,256 parameters and is the model used by `generate.py`.

## Temperature experiment

Generation was tested at temperatures `0.2`, `0.5`, `1.0`, and `2.0`. Temperature `0.5` gave the most useful balance of coherence and variety. Lower values make output more conservative; higher values make it more varied but less coherent.

## Installation

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
```

## Generate text

From this project directory:

```bash
python src/generate.py
```

The script loads `models/shakespeare_lstm_v2.keras` and prints samples for each tested temperature.

## Train from scratch

```bash
python src/train.py
```

Training downloads the corpus if needed, trains the V2 configuration, and writes the model plus training artifacts under `models/` and `outputs/`. Training can be memory- and compute-intensive because the full inputs are one-hot encoded.

## Limitations

- Character-level generation has no explicit word- or sentence-level understanding.
- One-hot encoding is memory-intensive.
- Output quality varies with the seed text and temperature.
- This is trained on a fixed Shakespeare corpus and does not provide factual or contextual guarantees.

## Future improvements

- Use embeddings to reduce memory use.
- Add validation data, checkpoints, and reproducible seeds.
- Compare GRU or Transformer-based baselines.
- Add a command-line interface for custom prompts, lengths, and temperatures.

## Technologies

Python, NumPy, TensorFlow/Keras, LSTM, and softmax sampling.
