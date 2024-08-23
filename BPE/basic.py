from tqdm import tqdm
from .base import Tokenizer


class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    

    def train(self, text, vocab_size, verbose=False):

        # Store the raw text
        self.raw_text = text
        
        # Set the desired final vocabulary size
        self.vocab_size = vocab_size
        
        # Calculate the number of merges to perform
        no_of_merges = vocab_size - 256  # Initially, we have 0...255 (256 vocabs)
        ids = self.getToken(text)  
        
        # Perform merges to create new tokens from top pairs
        for i in tqdm(range(no_of_merges)):
            stats = self.get_stats(ids)
            top_pair = max(stats, key=stats.get)
            idx = 256 + i  # Start assigning new tokens from 256 onwards
            
            # Merge the top pair into a new token
            ids = self.merge(ids, top_pair, idx)
            self.merges[top_pair] = idx
            self.vocab[idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            
            # a discription of merges while traning
            if verbose:
                print(f"merge {i+1}/{no_of_merges}: {top_pair} -> {idx} ({self.vocab[idx]}) had {stats[top_pair]} occurrences")

    
    def getToken(self, text):
        """
        Converts a string to a list of integers (tokens) by encoding it as UTF-8.
        """
        token = text.encode("utf-8")
        token = list(map(int, token))
        return token


    def encode(self, text):
        """
        Encodes a string into a list of integers (tokens) using the vocabulary.
        """
        tokens = self.getToken(text)
        
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            # extract the minimum extracted pair
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # Stop if there are no more pairs to merge
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens
    
    def decode(self, ids):
        """
        Decodes a list of integers (token ids) back into a string.
        """
        i = 0
        length = len(ids)
        
        # List out keys and values separately from the merges dictionary
        key_list = list(self.merges.keys())
        val_list = list(self.merges.values())
        
        while i < length:
            if ids[i] in val_list:
                position = val_list.index(ids[i])
                ids[i] = key_list[position][0]
                
                if i < len(ids) - 1:
                    ids.insert(i+1, key_list[position][1])
                else:
                    ids.append(key_list[position][1])
                
                i -= 1  # Move back to reprocess the new pair
                length = len(ids)  # Update the length after inserting
            i += 1
        
        # Decode the list of ids back into a UTF-8 string, replacing errors with '�'
        txt = bytes(ids).decode("utf-8", errors="replace")
        return txt


if __name__ == "__main__":
    print("hello")