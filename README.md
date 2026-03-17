# Word2Vec from scratch with NumPy

This project implements the Word2Vec algorithm (skip-gram with negative sampling) from scratch using only NumPy. No deep learning frameworks like PyTorch or TensorFlow were used — all forward passes, loss computations, gradient derivations, and parameter updates are implemented manually.

The goal of this project was to demonstrate understanding of the Word2Vec model, its training dynamics, and the ability to implement it using only basic numerical computing tools.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Requirements](#requirements)
- [Possible Improvements](#possible-improvements)
- [Author](#author)

## Overview

Word2Vec is a popular technique for learning word embeddings (dense vector representations of words) from large text corpora. The skip-gram model with negative sampling is one of its most efficient variants:

- **Skip-gram**: Predicts context words given a center word.
- **Negative Sampling**: Simplifies training by distinguishing the target context word from a small number of randomly sampled negative words.

This project implements:

- Text preprocessing (tokenization, vocabulary building, low-frequency filtering, removing punctuation and lowercasing)
- Generating center-context pairs
- Negative sampling using a smoothed unigram distribution (with power 0.75)
- Forward pass, loss computation, and gradient calculations
- Manual weight updates for input and output embeddings
- Saving/loading model weights and vocabulary
- Finding similar words and solving word analogies

## Project Structure
```txt
WORD2VEC-NUMPY/
├── notebooks/
│   ├── example.ipynb
│   └── notebook.ipynb
├── word2vec/
│   ├── __init__.py
│   ├── negative_sampler.py
│   ├── text_dataset.py
│   ├── vocabulary.py
│   └── word2vec_model.py
├── .gitignore
├── README.md
├── requirements.txt
└── train.py
```
## Dataset

The training script expects a plain text file at `word2vec/dataset.txt`.  
You can use any text dataset (e.g., Wikipedia dump, book corpus, news articles).  
During training, words that appear fewer than `min_count` times are filtered out to reduce noise and vocabulary size. The one used in to train my model was WikiText-2-raw-v1 from Hugging Face datasets containing raw data from encyclopedic articles.

## Training

Run the training script:

```bash
python train.py
```

Key training parameters (can be modified in train.py):

- epochs: 15

- window_size: 2

- embedding_dim: 30

- learning_rate: 0.01

- num_negative_samples: 5

- min_count: 10 (words with count < 10 are ignored)

## Evaluation

The Jupyter notebook example.ipynb demonstrates:

- Loading a trained model

- Finding the most similar words for a given query

- Solving word analogies (e.g., "king - man + woman = ?")

## Results

The model captures semantic relationships reasonably well, especially for frequent words with clear co-occurrence patterns. However, for less frequent words, the embeddings can be noisy or unexpected (examples included in the notebook). This may be due to the relatively small dataset and low number of training epochs.

## Usage

- Clone the repository.

- Install dependencies:
```bash
    pip install -r requirements.txt
```

- Prepare a text file at word2vec/dataset.txt.

- Run python train.py.

- Explore the results in example.ipynb.

## Requirements

- Python 3.7+

- NumPy

- datasets (only used for optional dataset loading in the notebook)

## Possible Improvements

- CBOW implementation: Adding an alternative training objective.

- Subsampling of frequent words: Speeding up training and improve embeddings.

- Learning rate scheduling: Decaying learning rate over time.

- More advanced evaluation: Test on standard word similarity benchmarks.

- Efficiency: Implement batched updates (currently processes one pair at a time).

## Author

This project was created as part of a coding challenge to demonstrate deep understanding of Word2Vec, gradient-based optimization, and the ability to implement complex models from scratch using only NumPy.
