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
    
    def forward_pass(self,pair):
        score = self.words_in[pair[0]] @ self.words_out[pair[1]]
        probability = self.sigmoid(score)
        return probability
    
    def compute_loss(self,positive_score,negative_scores):
        loss = np.log(self.sigmoid(positive_score)) + np.sum(np.log(self.sigmoid(-negative_scores)))
        return loss
    def training_step(self,pair,negatives):
        # (1−σ(score_pos))vo​−i=1∑k​(1−σ(−score_neg[i]))vn​[i]
        # compute scores for positive and negatives
        vector_word_in = self.words_in[pair[0]]
        vector_word_out = self.words_out[pair[1]]

        score_pos = vector_word_in @ vector_word_out

        vectors_neg = self.words_out[negatives]
        score_neg = vectors_neg @ vector_word_in

        # Compute gradients
        gradient_vector_center = (1-self.sigmoid(score_pos)) * vector_word_out - np.sum(1-self.sigmoid(-score_neg)[:,None]*vectors_neg,axis=0)
        gradient_vector_context = (1-self.sigmoid(score_pos)) * vector_word_in
        gradient_neg = -(1 - self.sigmoid(-score_neg))[:, None] * vector_word_in[None, :]

        # Update weights: center word, context word, negative words
        self.words_in[pair[0]] += self.learning_rate * gradient_vector_center

        self.words_out[pair[1]] += self.learning_rate * gradient_vector_context

        self.words_out[negatives] += self.learning_rate * gradient_neg


