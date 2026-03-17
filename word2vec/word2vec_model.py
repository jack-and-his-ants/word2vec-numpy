import numpy as np

class Word2VecModel():
    def __init__(self, vocab_size,embedding_dim=100,learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        limit = 0.5/embedding_dim
        self.words_in = np.random.uniform(-limit,limit,(vocab_size,embedding_dim))
        self.words_out = np.random.uniform(-limit,limit,(vocab_size,embedding_dim))

    def sigmoid(self,value):
        x = np.clip(value,-15,15)
        return 1/(1+np.exp(-x))
        
    def change_learning_rate(self, factor=0.1):
        self.learning_rate *= factor
        return self.learning_rate
    
    def forward_pass(self,pair):
        score = self.words_in[pair[0]] @ self.words_out[pair[1]]
        probability = self.sigmoid(score)
        return probability
    
    def compute_loss(self,positive_score,negative_scores):
        loss = (np.log(self.sigmoid(positive_score)) + np.sum(np.log(self.sigmoid(-negative_scores))))/(len(negative_scores)+1)
        return loss
    
    def training_step(self,pair,negatives):
        # compute scores for positive and negatives
        vector_word_in = self.words_in[pair[0]]
        vector_word_out = self.words_out[pair[1]]

        score_pos = vector_word_in @ vector_word_out

        vectors_neg = self.words_out[negatives]
        score_neg = vectors_neg @ vector_word_in

        # Compute gradients
        gradient_vector_center = ((1-self.sigmoid(score_pos)) * vector_word_out - np.sum((1-self.sigmoid(-score_neg))[:,None]*vectors_neg,axis=0))
        gradient_vector_context = (1-self.sigmoid(score_pos)) * vector_word_in
        gradient_neg = -(1 - self.sigmoid(-score_neg))[:, None] * vector_word_in[None, :]

        # Update weights: center word, context word, negative words
        self.words_in[pair[0]] += self.learning_rate * gradient_vector_center

        self.words_out[pair[1]] += self.learning_rate * gradient_vector_context
    
        for i, neg_idx in enumerate(negatives):
            self.words_out[neg_idx] += self.learning_rate * gradient_neg[i]

    def save(self, path):
        np.savez(
            path,
            words_in=self.words_in,
            words_out=self.words_out,
            embedding_dim=self.embedding_dim,
            vocab_size=self.vocab_size,
            learning_rate=self.learning_rate
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        vocab_size = int(data["vocab_size"])
        embedding_dim = int(data["embedding_dim"])
        learning_rate = float(data["learning_rate"])
        model = cls(vocab_size, embedding_dim, learning_rate)
        model.words_in = data["words_in"]
        model.words_out = data["words_out"]
        return model
    
    def get_k_closest_neighbours(self, word_index, k):

        v = self.words_in[word_index]

        scores = self.words_in @ v

        norms = np.linalg.norm(self.words_in, axis=1) * np.linalg.norm(v)
        similarities = scores / norms

        ind = np.argpartition(similarities, -(k+1))[-(k+1):]

        ind = ind[ind != word_index]

        return ind[:k]
    
    def analogy(self, word_a, word_b, word_c, vocab, k=5):
        #analogy = word_a - word_b + word_c

        idx_a = vocab.get_index_by_word(word_a.lower())
        idx_b = vocab.get_index_by_word(word_b.lower())
        idx_c = vocab.get_index_by_word(word_c.lower())

        v_a = self.words_in[idx_a]
        v_b = self.words_in[idx_b]
        v_c = self.words_in[idx_c]

        target = v_a - v_b + v_c

        scores = self.words_in @ target
        norms = np.linalg.norm(self.words_in, axis=1) * np.linalg.norm(target) + 1e-9
        similarities = scores / norms

        indices = np.argpartition(similarities, -(k+3))[-(k+3):]

        indices = indices[np.argsort(similarities[indices])[::-1]]

        result = []
        for idx in indices:
            if idx not in [idx_a, idx_b, idx_c]:
                result.append(vocab.get_word_by_index([idx])[0])
            if len(result) == k:
                break

        return result


