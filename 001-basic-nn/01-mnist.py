import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# Change the dataset name
dataset = "F_MNIST_data"
trained_dataset_name = "f_mnist_model.pth"

# step 1: load the data, normalize it
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

trainset =  datasets.MNIST('~/pytorch/'+dataset+'/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# step 2: define the model, the criterion (error function), optim (gradient function)
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1)
)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

# step 3: run the training in loops
epochs = 5
for e in range(epochs):
    runningloss = 0
    
    for images, labels in trainloader:
        optimizer.zero_grad()   #3a. clear the older grad values, bcs they accumulate

        images = images.view(images.shape[0], -1)   #3b. flatten the images

        output = model.forward(images)      #3c. forward pass to get the output
        loss = criterion(output, labels)    #3d. calculate loss from the output
        loss.backward()     #3e. run the backward pass, calc the gradient values
        optimizer.step()    #3f. apply the gradient descent and update the weights

        runningloss += loss.item()
        print(f"item_loss: {loss.item()}, runningloss: {runningloss}")
    else:
        print(f"training loss: {runningloss / len(trainloader)}")

# step 4: save the trained model
torch.save(model.state_dict(), trained_dataset_name)
print("Model saved!")
