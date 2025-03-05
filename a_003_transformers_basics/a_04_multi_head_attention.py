from a_03_attention_block import AttentionHead
import torch.nn as nn
import torch
from pathlib import Path
from a_01_tokenizer import CharTokenizer


###### Main class #########
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # create a list of attentionheads, each initialized with the config param, length = config["heads_num"]
        heads_list = [AttentionHead(config) for _ in range(config["heads_num"])]

        # store that created list in ModuleList function
        self.heads = nn.ModuleList(heads_list)

        # create a linear layer with size embedding_dim * embedding_dim
        self.linear = nn.Linear(config["embedding_dim"], config["embedding_dim"])

        # create a dropout layer with rate = config["dropout_rate"], store it in self.dropout
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, input):
        # pass the input through each head and store the output as a list
        heads_output = [head(input) for head in self.heads]

        # concatenate all heads, store it in scores_change matrix
        scores_change = torch.cat(heads_output, dim=-1)

        # pass that scores_change through the linear layer
        scores_change = self.linear(scores_change)

        # return the final result by passing it through dropout
        return self.dropout(scores_change)
    
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

####### Runner code #######
input = torch.rand(8, config["context_size"], config["embedding_dim"])
mha = MultiHeadAttention(config)
output = mha(input)
print(f"Output size of the multiple attention heads: {output.shape}")