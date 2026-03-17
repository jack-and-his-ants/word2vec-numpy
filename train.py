import numpy as np
from word2vec import TextDataset, NegativeSampler, Word2VecModel

# ======================
# Loading data
# ======================
with open("word2vec/dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

# we choose numbers only with count >= 10
dataset = TextDataset(text, min_count=10)

# generating center-context pairs with window size 2
pairs = dataset.generate_pairs(window_size=2)

print(f"Vocabulary size: {dataset.vocab_size}")
print(f"Number of pairs: {len(pairs)}")

# ======================
# training parameters
# ======================
epochs = 15
num_negative_samples = 5
embedding_dim = 30
learning_rate = 0.01

# ======================
# Creating model and negative sampler
# ======================
model = Word2VecModel(dataset.vocab_size, embedding_dim, learning_rate)
sampler = NegativeSampler(dataset.counts)

# ======================
# Training
# ======================
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    np.random.shuffle(pairs)

    # Pre-sample negative words for an entire epoch
    negatives = sampler.sample(num_negative_samples * len(pairs))

    for num, (center, context) in enumerate(pairs):
        neg_sample = negatives[num_negative_samples*num:num_negative_samples*num + num_negative_samples]
        model.training_step((center, context), neg_sample)

# ======================
# Save model
# ======================
model.save("word2vec_model.npz")
dataset.save_vocab('word2vec_vocabulary.pkl')
print("Model saved to word2vec_model.npz and vocabulary saved to word2vec_vocabulary.pkl")