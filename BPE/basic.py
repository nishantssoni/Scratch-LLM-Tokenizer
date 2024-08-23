from tqdm import tqdm
from .base import Tokenizer


class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    

    def train(self, text, vocab_size, verbose=False):

        # Store the raw text
        self.raw_text = text
        
        # Tokenize the text and calculate statistics for token pairs
        # self.tokens = self.getToken(self.raw_text)
        # self.stats = self.get_stats(self.tokens)
        
        # Set the desired final vocabulary size
        self.vocab_size = vocab_size
        
        # Calculate the number of merges to perform
        no_of_merges = vocab_size - 256  # Initially, we have 0...255 (256 vocabs)
        ids = self.getToken(text)  # Copy tokens to ids so we don't modify the original list
        
        # Perform merges to create new tokens from top pairs
        for i in tqdm(range(no_of_merges)):
            stats = self.get_stats(ids)
            top_pair = max(stats, key=stats.get)
            idx = 256 + i  # Start assigning new tokens from 256 onwards
            
            # Merge the top pair into a new token
            ids = self.merge(ids, top_pair, idx)
            self.merges[top_pair] = idx



if __name__ == "__main__":
    print("hello")