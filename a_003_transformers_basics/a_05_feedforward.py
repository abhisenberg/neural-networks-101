import torch 
import torch.nn as nn
from pathlib import Path
from a_01_tokenizer import CharTokenizer


###### Main class #########
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # create a sequential nn with the following layers
        # 1. expand the input size to 4 times
        # 2. pass it through gelu
        # 3. contract the output size back to original
        # 4. pass it through dropout layer

        self.linear_layers = nn.Sequential(
            nn.Linear(config["embedding_dim"], config["embedding_dim"]*4),
            nn.GELU(),
            nn.Linear(config["embedding_dim"]*4, config["embedding_dim"]),
            nn.Dropout(config["dropout_rate"])
        )

    def forward(self, input):
        return self.linear_layers(input)
    
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
ff = FeedForward(config)
inp = torch.rand(8, config["context_size"], config["embedding_dim"])
output = ff(inp)
print(output.shape)