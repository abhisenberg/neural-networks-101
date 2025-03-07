import torch
import torch.nn as nn
from a_06_transformer_block import TransformerBlock
from a_01_tokenizer import CharTokenizer
from pathlib import Path
import torch.nn.functional as F

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

######## Main class ##########

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DemoGPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        # create an embedding layer, token_embedding_layer with size vocab_size -> embedding_dim
        # create an embedding layer, positional_embedding_layer with size context_size -> embedding_dim
        self.token_embedding_layer = nn.Embedding(config["vocabulary_size"], config["embedding_dim"])
        self.positional_embedding_layer = nn.Embedding(config["context_size"], config["embedding_dim"])

        # create a set of blocks, create a sequential layer using those blocks
        blocks = [TransformerBlock(config) for _ in range(config["layers_num"])]

        self.transformer_block_layers = nn.Sequential(*blocks)
        self.layer_norm = nn.LayerNorm(config["embedding_dim"])
        self.unembedding = nn.Linear(config["embedding_dim"],  config["vocabulary_size"], bias=False)

    def forward(self, token_ids):
        batch_size, tokens_num = token_ids.shape
        
        # we'll generate the token_embedding as well as positional_embedding
        token_embedding = self.token_embedding_layer(token_ids)
        sequence = torch.arange(tokens_num, device=device)
        positional_embedding = self.positional_embedding_layer(sequence)

        # we'll then combine the 2 to get a proper input containing info about "what" the token is, and "where" it appears in the sequence
        x = token_embedding + positional_embedding

        # pass the input through the transformer blocks, then normalisation layer, then convert model's internal representation back to logits over the vocab 
        x = self.transformer_block_layers(x)
        x = self.layer_norm(x)
        x = self.unembedding(x)
        return x

###### Runner Code ########
model = DemoGPT(config).to(device)
output = model(tokenizer.encode("hi").unsqueeze(dim=0).to(device))
print(output.shape)

def generate(model, prompt_ids, max_tokens):

    # The function initializes output_ids with the prompt and then iteratively generates max_tokens new tokens, one at a time.
    output_ids = prompt_ids
    for _ in range(max_tokens):

        # This stops generation if the sequence exceeds the model's maximum context size (the maximum sequence length the model was designed to handle).
        if output_ids.shape[1] >= config["context_size"]:
            break
        
        # Disables gradient tracking to save memory during inference. The model processes the entire sequence so far and returns logits (raw, unnormalized predictions)
        with torch.no_grad():
            logits = model(output_ids)
        
        # This selects only the logits for the last token in the sequence, as we only need to predict what comes after the current sequence.
        logits = logits[:, -1, :]

        # Applies softmax to convert raw logits into a probability distribution over the entire vocabulary.
        probs = F.softmax(logits, dim=-1)
        
        # sample a random token given the softmax distribution
        """
        torch.multinomial is a PyTorch function that samples from a probability distribution. It's like rolling a weighted dice:

        It takes a tensor of probabilities (probs) that sum to 1
        It randomly selects elements according to those probabilities
        The higher the probability, the more likely that element will be chosen
        """
        next_token_id = torch.multinomial(probs, num_samples=1)

        # add new token to the output, repeat the process
        output_ids = torch.cat([output_ids, next_token_id], dim=-1)

    return output_ids

def generate_with_prompt(model, tokenizer, prompt, max_tokens=1000):
    model.eval()
    prompt = tokenizer.encode(prompt).unsqueeze(dim=0).to(device)
    response_tokens = generate(model, prompt, max_tokens=max_tokens)
    return tokenizer.decode(response_tokens[0])


generated_prompt = generate_with_prompt(model, tokenizer, "First Citizen:\n")
print(generated_prompt)
