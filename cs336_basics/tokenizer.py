import json
import regex as re
from typing import Iterable, Iterator
import sys

from tests.common import gpt2_bytes_to_unicode

INT_MAX = 2147483647

class Tokenizer:
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab:dict[int, bytes] = vocab
        self.merges:list[tuple[bytes, bytes]] = merges
        self.special_tokens = special_tokens
        self.revocab={self.vocab[word]:word for word in self.vocab}
        self.bpe_ranks = {merge_pair: rank for rank, merge_pair in enumerate(self.merges)}


    @classmethod
    def from_files(cls, 
                   vocab_filepath, 
                   merges_filepath, 
                   special_tokens=None):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        IDs:list[int] = []

        #0.使用特殊token分块

        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(tok) for tok in sorted_special_tokens]
            split_pattern = "(" + "|".join(escaped_tokens) + ")"
            chunks = re.split(split_pattern, text)
        else:
            chunks = [text]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in chunks:
            if not chunk:
                continue
            
            if self.special_tokens and chunk in self.special_tokens:
                IDs.append(self.revocab[chunk.encode("utf-8")])
                continue
            else:
                for match in re.finditer(PAT, chunk):
                    word = tuple(match.group().encode("utf-8")[i : i+1] for i in range(len(match.group().encode("utf-8"))))
                    
                    #1.合并直到无法合并
                    while True:
                        #1.寻找优先级最高的合并组合
                        couples = list(zip(word[:-1],word[1:]))
                        most_common_couple = ()
                        priority = INT_MAX
                        for couple in couples:
                            if couple in self.bpe_ranks and self.bpe_ranks[couple]<priority:
                                most_common_couple = couple
                                priority = self.bpe_ranks[couple]
                        if most_common_couple == (): break

                        #2.合并
                        new_word:list[bytes]=[]
                        new_word.append(word[0])
                        last_merged = False
                        for pos in range(1,len(word)):
                            if last_merged == True:
                                last_merged = False
                                new_word.append(word[pos])
                                continue
                            if(word[pos-1],word[pos])==most_common_couple:
                                new_word.append(new_word.pop()+word[pos])
                                last_merged = True
                            else:
                                new_word.append(word[pos])
                        word = new_word
                    
                    #2.查反向表
                    for element in tuple(word):
                        IDs.append(self.revocab[element])
        return IDs

    def decode(self, ids: list[int]) -> str:
        text = b""
        for i in ids:
            text += self.vocab[i]
        return text.decode("utf-8",errors="replace")
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for id in self.encode(text):
                yield(id)

if __name__ == "__main__":
    with open ("/home/chino/prog/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt",'r') as f:
        text = f.read()
    t = Tokenizer.from_files("/home/chino/prog/assignment1-basics/cs336_basics/vocab_tiny.txt","cs336_basics/merges_tiny.txt",["<|endoftext|>"])
    #print(t.decode(t.encode(text)))
    encoder = t.encode_iterable([text])
    for i in encoder:
        print(i)
        print(t.decode([i]))
