from .base import Tokenizer
import regex as re
from tqdm import tqdm

class RgxTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        

    def train(self, text, v_size, verbose=False):
        """
        Generates the vocabulary from the raw text based on the desired vocabulary size.
        """
        # Store the raw text
        self.raw_text = text
        
        
        # Set the desired final vocabulary size
        self.vocab_size = v_size
        
        # Calculate the number of merges to perform
        num_merges = v_size - 256  # Initially, we have 0...255 (256 vocabs)
        ids = self.getToken(text)  # Copy tokens to ids so we don't modify the original list

        merges = {}
        
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # Perform merges to create new tokens from top pairs
        for i in tqdm(range(num_merges)):
            stats = {}
            for chunk_ids in ids:
                self.get_stats(chunk_ids, stats)
            top_pair = max(stats, key=stats.get)
            idx = 256 + i  # Start assigning new tokens from 256 onwards
            
            # Merge the top pair into a new token
            ids = [self.merge(chunk_ids, top_pair, idx) for chunk_ids in ids]
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            
            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {idx} ({vocab[idx]}) had {stats[top_pair]} occurrences")
        
        
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def return_regex_token(self, text, ver="GPT4"):
        if ver == "GPT4":
            pttrn = re.compile(self.GPT4_SPLIT_PATTERN)
        else:
            pttrn = re.compile(self.GPT2_SPLIT_PATTERN)

        text_chunks = re.findall(pttrn, text)
        
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        
        return ids    
    
    
    def getToken(self, text, ver="GPT4"):
        """
        Converts a string to a list of integers (tokens) by encoding it as UTF-8.
        """
        if ver == "GPT4":
            pttrn = re.compile(self.GPT4_SPLIT_PATTERN)
        else:
            pttrn = re.compile(self.GPT2_SPLIT_PATTERN)

        text_chunks = re.findall(pttrn, text)
        
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        
        return ids



    def encode_in_chunk(self, text):
        """
        Encodes a string into a list of integers (tokens) using the vocabulary.
        """
        # tokens = list(text.encode("utf-8"))
        tokens = self.getToken(text)
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # Stop if there are no more pairs to merge
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens
    
    def encode(self, text):
        pttrn = re.compile(self.GPT4_SPLIT_PATTERN)
        text_chunk = re.findall(pttrn, text)

        ids = []
        for i in text_chunk:
            temp_ids = self.encode_in_chunk(i)
            ids.extend(temp_ids)
        print("i am encode ids :: ", ids)
        return ids


    def decode(self, ids):
        """
        Decodes a list of integers (token ids) back into a string.
        """
        i = 0
        ids = [item for sublist in ids for item in sublist]
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
        
        # print("the ids are :: ",ids)
        # Decode the list of ids back into a UTF-8 string, replacing errors with '�'
        txt = bytes(ids).decode("utf-8", errors="replace")
        return txt
        
    def print_vocab(self, start=0, end=0):
        """
        Prints the vocabulary from start to end index.
        """
        if end == 0:
            end = self.vocab_size
        for i, item in enumerate(list(self.vocabs.items())[start:end]):
            print(item)
    
    def print_stats(self):
        """
        Prints the token pair statistics sorted by frequency.
        """
        a = sorted(((v, k) for k, v in self.stats.items()), reverse=True)
        for i in a:
            print(i)
    
    def validate(self, text):
        """
        Validates the encoding and decoding process by checking if the decoded text matches the original.
        """
        decoded_txt = self.decode(self.encode(text))
        return (text == decoded_txt)
    
    def write_to_file(self):
        with open('output_vocab_regex.txt', 'w', encoding='utf-8') as file:
            for i, item in enumerate(list(self.vocabs.items()), start=1):
                file.write(f"{i}. {item[1]}\n")
        
        with open('output_merges_regex.txt', 'w', encoding='utf-8') as file:
            for i, item in enumerate(list(self.merges.items()), start=1):
                file.write(f"{i}::  '{self.decode(list(item[0]))}', {item[1]}\n")