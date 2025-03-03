import torch
import torch.nn as nn
from pathlib import Path
from a_01_tokenizer import CharTokenizer
from a_02_data_loader import TokenIDsDataset

text = Path('../training-data/tiny-shakespeare.txt').read_text()

tokenizer = CharTokenizer.trainFromText(text)

print(tokenizer.encode("hello world"))
print(tokenizer.decode(tokenizer.encode("hello world")))

# config = {
#     "vocabulary_size": tokenizer.vocabSize(),   #size of vocab / # of unique token IDs that it supports
#     "context_size": 256,    #lentgh of the context window / max number of tokens the model can see at once
#     "embedding_dim": 768,   #length of the embedding vector, each token will be converted to an embedding vector of this length
#     "heads_num": 12,        #num of attention heads we'll have in the model, each of them will process the input independently
#     "layers_num": 10,       #num of layers / transformers blocks in the model
#     "dropout_rate": 0.1,    #probability of dropping out nodes on random
#     "use_bias": False,      #whether the linear transformations should include bias terms
# }

config = {
    "vocabulary_size": tokenizer.vocabSize(),   #size of vocab / # of unique token IDs that it supports
    "context_size": 32,    #lentgh of the context window / max number of tokens the model can see at once
    "embedding_dim": 64,   #length of the embedding vector, each token will be converted to an embedding vector of this length
    "heads_num": 4,        #num of attention heads we'll have in the model, each of them will process the input independently
    "layers_num": 2,       #num of layers / transformers blocks in the model
    "dropout_rate": 0.1,    #probability of dropping out nodes on random
    "use_bias": False,      #whether the linear transformations should include bias terms
}

# we need to set the size of output form each head correctly
# so that in the end when we merge the output from all heads,
# the final vector will be of the right size
config["head_size"] = config["embedding_dim"] // config["heads_num"]

# 1. let's first implement a single attention head
class AttentionHead(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.WQ = nn.Linear(config['embedding_dim'], config['head_size'], config['use_bias'])
        self.WK = nn.Linear(config['embedding_dim'], config['head_size'], config['use_bias'])
        self.WV = nn.Linear(config['embedding_dim'], config['head_size'], config['use_bias'])

        self.dropout = nn.Dropout(config["dropout_rate"])

        causal_attention_mask = torch.tril(torch.ones(config['context_size'], config['context_size']))
        self.register_buffer("causal_attention_mask", causal_attention_mask)

        print(f"WQ: {self.WQ}, WK: {self.WK}, Wv: {self.WV}")

    def forward(self, input):   # (B, C, embedding_dim)
        batch_size, tokens_num, embedding_dim = input.shape
        Q = self.WQ(input)  # (B, C, head_size)
        K = self.WK(input)  # (B, C, head_size)
        V = self.WV(input)  # (B, C, head_size)

        print(f"WQ: {self.WQ}, Q: {Q}")
        print(f"WK: {self.WK}, K: {K}")
        print(f"WV: {self.WV}, V: {V}")

        attention_scores = Q @ K.transpose(1,2) # (B, C, C)

        print(f"1. Att scores: {attention_scores}")

        attention_scores = attention_scores.masked_fill(
            self.causal_attention_mask[:tokens_num, :tokens_num] == 0,
            -torch.inf
        )

        print(f"2. Att scores: {attention_scores}")

        attention_scores = attention_scores / (K.shape[-1] ** 0.5)
        print(f"3. Att scores: {attention_scores}")
        attention_scores = torch.softmax(attention_scores, dim=-1)
        print(f"4. Att scores: {attention_scores}")
        attention_scores = self.dropout(attention_scores)
        print(f"5. Att scores: {attention_scores}")

        return attention_scores @ V # (B, C, head_size)

# 2. testing the attention head with a random input
"""
Here,
batch_size = 8
context_size = 256
embedding_dim = 768
"""
inp = torch.rand(1, config['context_size'], config['embedding_dim'])

ah = AttentionHead(config)
output = ah(inp)
print(output.shape)