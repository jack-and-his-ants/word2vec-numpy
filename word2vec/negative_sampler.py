import numpy as np

class NegativeSampler():
    def __init__(self,words_counts,table_size = 10000000):
        counts_float = np.copy(words_counts)
        self.vocab_size = len(words_counts)
        probs = words_counts.astype(np.float64) ** 0.75
        probs /= probs.sum()

        counts = np.round(probs * table_size).astype(int)

        table = []

        for word_index, c in enumerate(counts):
            table.extend([word_index] * c)

        table = np.array(table, dtype=np.int32)

        if len(table) > table_size:
            table = table[:table_size]
        elif len(table) < table_size:
            extra = np.random.choice(table, table_size - len(table))
            table = np.concatenate([table, extra])

        self.table = table
        self.table_size = len(table)

    def sample(self,k,forbidden_word=None):
        idx = np.random.randint(0, self.table_size, size=k)
        return self.table[idx]
