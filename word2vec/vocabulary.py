import pickle

class Vocabulary():
    def __init__(self,word_to_index,index_to_word):
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        
    def get_word_by_index(self, index):
        words= []
        for i in index:
            words.append(self.index_to_word[i])
        return words
    
    def get_index_by_word(self,word):
        return self.word_to_index[word]
    
    def save(self,path):
        vocab_data = {
        "word_to_index": self.word_to_index,
        "index_to_word": self.index_to_word
        }
        with open(path, "wb") as f:
            pickle.dump(vocab_data, f)

    def __len__(self):
        return len(self.index_to_word)
        
    @classmethod
    def load_dict(cls,path):
        with open(path, "rb") as f:
            vocab_data = pickle.load(f)
        vocab = cls(vocab_data["word_to_index"],vocab_data['index_to_word'])
        return vocab 