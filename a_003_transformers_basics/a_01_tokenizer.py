import torch
from pathlib import Path

class CharTokenizer:
    def __init__(self, vocab):
        self.char_to_tokenid = { char:index  for index, char in enumerate(vocab)}
        self.tokenid_to_char = { index:char  for index, char in enumerate(vocab)}

    @staticmethod
    def trainFromText(text):
        vocab = sorted(list(set(text)))
        return CharTokenizer(vocab)

    def encode(self, text):
        return torch.tensor([self.char_to_tokenid[ch] for ch in text])

    def decode(self, tokenids):
         return ''.join([self.tokenid_to_char[token] for token in tokenids.tolist()])

    def vocabSize(self):
        return len(self.char_to_tokenid)


# Runner code
text = Path('../training-data/tiny-shakespeare.txt').read_text()
print(text[:1000])

ct = CharTokenizer.trainFromText(text)

a = "hello world!"
encoded = ct.encode(a)
decoded = ct.decode(encoded)
print(encoded)
print(decoded)