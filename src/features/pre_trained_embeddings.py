import numpy as np

class PreTrainedEmbeddings(object):
    def __init__(self, word_to_index, word_vectors):
        self.word_to_index = word_to_index
        self.word_vectors = word_vectors
        self.index_to_word = {v : k for k,v in self.word_to_index.items()}
        
    @classmethod
    def from_embeddings_file(cls, embedding_file):
        word_to_index = {}
        word_vectors = []
        with open(embedding_file) as fp:
            for line in fp.readlines():
                line = line.split(" ")
                word = line[0]
                vec = np.array([float(x) for x in line[1:]])
                word_to_index[word] = len(word_to_index)
                word_vectors.append(vec)
        return cls(word_to_index, word_vectors)
    
    def get_embedding(self, word):
        return self.word_vectors[self.word_to_index[word]]