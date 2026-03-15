import numpy as np

class NegativeSampler():
    def __init__(self,words_counts):
        counts_float = np.copy(words_counts)
        self.probabilities = counts_float.astype(float) ** 0.75
        self.probabilities /= np.sum(self.probabilities)
        self.vocab_size = len(self.probabilities)

    def sample(self,k,forbidden_word=None):
        choices = np.random.choice(self.vocab_size, k, p=self.probabilities)
        i=0
        while i < k:
            if choices[i] == forbidden_word:
                choices[i] = np.random.choice(self.vocab_size,p=self.probabilities)
            else:
                i+=1
        return choices
