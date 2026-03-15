import numpy as np
import math

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
        self.learning_rate /= factor
        return self.learning_rate
    
# model = Word2VecModel(10,2)
# print(model.words_in)