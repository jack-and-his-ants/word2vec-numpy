import re
from collections import Counter
from Vocabulary import Vocabulary
import numpy as np

class TextDataset():
    def __init__(self,text,min_count=1):
        self.tokens = self.tokenize(text)
        word_to_index,index_to_word,self.counts = self.build_vocabulary(min_count)

        self.vocabulary = Vocabulary(word_to_index,index_to_word)

        self.encoded_tokens = self.encode_tokens()
        self.vocab_size = len(self.vocabulary)
    
    def tokenize(self,text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", '',text)
        return text.split()
    
    def build_vocabulary(self,min_count=1):
        counts_dict = Counter(self.tokens)

        vocab = [word for word,count in counts_dict.items() if count>=min_count]

        word_to_idx = {word:index for index,word in enumerate(vocab)}

        idx_to_word = {index:word for word,index in word_to_idx.items()}

        counts = np.zeros(shape=(len(vocab)),dtype=np.int32)

        for word,count in counts_dict.items():
            if(count>=min_count):
                counts[word_to_idx[word]] = count

        return word_to_idx,idx_to_word,counts
    
    def encode_tokens(self):
        return np.array([self.vocabulary.word_to_index[word] for word in self.tokens if word in self.vocabulary.word_to_index],dtype=np.int32)

    def generate_pairs(self,window_size=2):
        pairs = []
        etokens = self.encoded_tokens
        for center_index,center in enumerate(etokens):
            end = min(center_index+window_size+1,len(etokens))
            start = max(0,center_index-window_size)
            for context_index in range(start,end):
                if center_index!=context_index:
                    context = etokens[context_index]
                    pairs.append([center,context])
        return np.array(pairs)
    def save_vocab(self, path):
        self.vocabulary.save(path)
    
    def get_word_by_idx(self, idx):
        return self.vocabulary.get_word_by_index(idx)
    def get_idx_by_word(self, word):
        return self.vocabulary.get_index_by_word(word)

    


if __name__ == "__main__":
    data = TextDataset("one, two, three, four, five")
    print(data.vocab_size)
    print(data.tokens)
    print(data.encoded_tokens)
    print(type(data.encoded_tokens))
    pairs = data.generate_pairs(2)
    for center,context in pairs:
        print(data.get_word_by_idx(center),data.get_word_by_idx(context))
