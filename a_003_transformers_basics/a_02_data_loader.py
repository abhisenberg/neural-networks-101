from torch.utils.data import Dataset
from pathlib import Path
from a_01_tokenizer import CharTokenizer

class TokenIDsDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # We can only start the sequences till the index where
        # we would have enough tokens available to generate the
        # output sequence as well
        return len(self.data) - self.block_size

    def __getitem__(self, pos):
        assert pos < len(self.data) - self.block_size

        # Here, x represents the input seq and y represents the output seq
        # Ex: x = [A, B, C], y = [B, C, D]
        # Meaning, if given a sequence like A,B,C. the next predicted item would be D 
        x = self.data[pos:pos + self.block_size]
        y = self.data[pos + 1:pos + 1 + self.block_size]
        return x, y
    

"""
We will import the text data and it will serve 2 purposes:
- Creating the tokenizer
- Creating the dataset for training
"""
text = Path('../training-data/tiny-shakespeare.txt').read_text()

# 1. Creating a tokenizer out of our text data
ct = CharTokenizer.trainFromText(text)

# 2. Creating the dataset for training by first encoding the data into tokens
encoded = ct.encode(text)

# 3. Create the dataset class
block_size = 64
dataset = TokenIDsDataset(data=encoded, block_size=block_size)  

# 4. Get the first item from the dataset and decode it
x, y = dataset[0]
print("X: ", ct.decode(x))
print("Y: ", ct.decode(y))
