import torch
import torch.nn as nn
from a_04_multi_head_attention import MultiHeadAttention
from a_05_feedforward import FeedForward
from pathlib import Path
from a_01_tokenizer import CharTokenizer

####### Main class ########
class TransformerBlock(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        # set the following elements in the self
        # - multi head attention block
        # - normalisation layer 1
        # - feedforward layer
        # - normalisation layer 2
        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config["embedding_dim"])
        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config["embedding_dim"])

    # Flow of the data is as:
    # LayerNorm1 -> MultiHeadAtt -> ResidualConnection1
    # LayerNorm2 -> FeedForward -> ResidualConnection2
    def forward(self, input):
        residual = input
        x = self.layer_norm_1(input)
        x = self.multi_head(x)
        x = x + residual

        residual = x
        x = self.layer_norm_2(input)
        x = self.feed_forward(x)
        x = x + residual

        return x

####### Utility variables #######
text = Path('../training-data/tiny-shakespeare.txt').read_text()
tokenizer = CharTokenizer.trainFromText(text)

config = {
    "vocabulary_size": tokenizer.vocabSize(),   #size of vocab / # of unique token IDs that it supports
    "context_size": 32,    #lentgh of the context window / max number of tokens the model can see at once
    "embedding_dim": 64,   #length of the embedding vector, each token will be converted to an embedding vector of this length
    "heads_num": 4,        #num of attention heads we'll have in the model, each of them will process the input independently
    "layers_num": 2,       #num of layers / transformers blocks in the model
    "dropout_rate": 0.1,    #probability of dropping out nodes on random
    "use_bias": False,      #whether the linear transformations should include bias terms
}
config["head_size"] = config["embedding_dim"] // config["heads_num"]

####### Runner code ########
tb = TransformerBlock(config)
input = torch.rand(8, config["context_size"], config["embedding_dim"])
output = tb(input)
print(output.shape)