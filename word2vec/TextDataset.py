import re
from collections import Counter

class TextDataset():
    def __init__(self,text,min_count=1):
        self.tokens = self.tokenize(text)
        self.wordToIndex,self.indexToWord,self.counts = self.buildVocabulary(min_count)
        self.encodedTokens = self.encodeTokens()
    
    def tokenize(self,text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", '',text)
        return text.split()
    
    def buildVocabulary(self,min_count=1):
        counts = Counter(self.tokens)
        vocab = [word for word,count in counts.items() if count>=min_count]
        print(vocab)
        wordToIdx = {word:index for index,word in enumerate(vocab)}
        idxToWord = {index:word for word,index in wordToIdx.items()}
        return wordToIdx,idxToWord,counts
    
    def encodeTokens(self):
        return [self.wordToIndex[word] for word in self.tokens if word in self.wordToIndex]



data = TextDataset("Test data test, yellow duck, purple duck, orange")
print(data.tokens)
print(data.encodedTokens)