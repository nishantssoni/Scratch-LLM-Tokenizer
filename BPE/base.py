class Tokenizer:
    """Base class"""

    def __init__(self) -> None:
        self.merges = {}
        self.pattern = ""
        self.secial_tokens = {}
        self.vocab_size = 0
        self.vocab = self.get_vocab()
        self.raw_text = ''

    
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError
    
    def encode(self, text):
        raise NotImplementedError
    
    def decode(self, ids):
        raise NotImplementedError
    
    def get_token(self, text):
        raise NotImplementedError
    
    def get_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        for (pair_1, pair_2), idx in self.merges.items():
            vocab[idx] = vocab[pair_1] + vocab[pair_2]
        
        return vocab
    
    def get_stats(self, ids, counter=None):
        counter = {} if counter is None else counter
        for pairs in zip(ids, ids[1:]):
            counter[pairs] = counter.get(pairs, 0) + 1
        return counter
    
    def merge(self, ids, pair, idx):
            """
            Merges a given pair of tokens in the ids list into a new token.
            
            ids: the list of token ids
            pair: the token pair to be merged
            idx: the integer representing the new token
            """
            i = 0
            length = len(ids)
            while i < (length - 1):
                if (ids[i] == pair[0]) and (ids[i+1] == pair[1]):
                    ids[i] = idx  # Replace the pair with the new token
                    ids.pop(i+1)  # Remove the second token of the pair
                    length = len(ids)  # Update the length after popping
                i += 1
            return ids