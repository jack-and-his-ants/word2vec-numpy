import numpy as np
from TextDataset import TextDataset
from NegativeSampler import NegativeSampler
from Word2VecModel import Word2VecModel
from Vocabulary import Vocabulary
text = """
The king loves the queen. The queen loves the king. The king rules the kingdom. 
The queen wears a crown. The man walks in the park. The woman walks in the park. 
The boy plays football. The girl plays football. The boy kicks the ball. 
The girl throws the ball. The dog chases the cat. The cat climbs the tree. 
The dog sleeps in the house. The cat watches the dog. The girl plays with the dog. 
The boy chases the cat. The man and the woman walk together. The boy runs fast. 
The girl runs fast. The dog runs after the ball. The cat jumps on the table. 
The king and the queen sit on the throne. The man reads a book. The woman reads a book. 
The boy reads a story. The girl reads a story. The dog barks loudly. The cat hisses. 
The boy eats an apple. The girl eats an apple. The man drinks water. The woman drinks water. 
The dog drinks water. The cat drinks milk. The boy writes a letter. The girl writes a letter. 
The man writes a letter. The woman writes a letter. The boy draws a picture. The girl draws a picture. 
The dog plays in the garden. The cat sleeps on the sofa. The man plays chess. The woman plays chess. 
The boy plays football in the park. The girl plays football in the park. The dog chases the ball. 
The cat chases the mouse. The king and queen watch the parade. The man and woman watch the parade. 
The boy and girl watch the parade. The dog watches the parade. The cat watches the parade.
"""

dataset = TextDataset(text,2)
pairs = dataset.generate_pairs(2)
epochs = 75
num_negative_samples = 5
d=(dataset.get_idx_by_word(input()))
model = Word2VecModel(dataset.vocab_size,30)
sampler = NegativeSampler(dataset.counts)
for epoch in range(epochs):
    combined_loss = 0.0
    np.random.shuffle(pairs)
    for center,context in pairs:
        negatives = sampler.sample(num_negative_samples,forbidden_word=context)
        model.training_step((center,context),negatives)
    
print(dataset.get_word_by_idx(model.get_k_closest_neighbours(d,5)))


