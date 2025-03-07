import a_02_data_loader
import torch
import torch.nn as nn
from a_06_transformer_block import TransformerBlock
from a_01_tokenizer import CharTokenizer
from a_02_data_loader import TokenIDsDataset
from a_07_gpt_class import DemoGPT
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from a_07_gpt_class import generate_with_prompt

####### Utility variables #######
text = Path('../training-data/tiny-shakespeare.txt').read_text()
tokenizer = CharTokenizer.trainFromText(text)

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

######## Defining model ####### 
model = DemoGPT(config).to(device)

######## Training Parameters ####### 
batch_size = 64
train_iterations = 5000
evaluation_interval = 100
learning_rate = 4e-4

train_data = tokenizer.encode(text).to(device)
train_dataset = TokenIDsDataset(train_data, config["context_size"])

train_sampler = RandomSampler(train_dataset, num_samples=batch_size * train_iterations, replacement=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

# optimizer for BP param updation
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for step_num, sample in enumerate(train_dataloader):
    model.train()
    input, targets = sample
    
    logits = model(input)

    """
    logits_view and targets_view are reshaped versions of the model outputs and target labels:
    1. Reshaping Purpose: They convert the 3D tensors (batch_size x sequence_length x vocabulary_size) into 2D tensors to match the expected input format for the loss function.
    2. Original Shapes:
        logits: [batch_size, context_size, vocabulary_size] - The raw model predictions
        targets: [batch_size, context_size] - The ground truth tokens
    3. Reshaped as:
        logits_view: [batch_size * context_size, vocabulary_size] - Flattening the batch and sequence dimensions
        targets_view: [batch_size * context_size] - Corresponding flattened targets
    """

    logits_view = logits.view(batch_size * config["context_size"], config["vocabulary_size"])
    targets_view = targets.view(batch_size * config["context_size"])

    loss = F.cross_entropy(logits_view, targets_view)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    print(f"Step: {step_num}, Loss: {loss.item()}:.3f")

    if step_num % evaluation_interval == 0:
        print("Demo GPT:\n" + generate_with_prompt(model, tokenizer, "\n"))