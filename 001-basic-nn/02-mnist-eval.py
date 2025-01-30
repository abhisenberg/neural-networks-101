import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

dataset = "F_MNIST_data"
trained_dataset_name = "f_mnist_model.pth"

# step 1: load the data, normalize it
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

trainset =  datasets.MNIST('~/pytorch/'+dataset+'/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# step 2: create the loaded model, load the training weights
loaded_model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1)
)

loaded_model.load_state_dict(torch.load(trained_dataset_name))
loaded_model.eval()

# step 3: evaluate with one image
with torch.no_grad():   #we don't need gradient for evaluation
    images, labels = next(iter(trainloader))    
    img = images[0].view(1, 784)    #get one img from the batch
    label = labels[0]
    logits = loaded_model.forward(img)
    print(torch.exp(logits), label)

# Test with one batch
with torch.no_grad():
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)
    ps = torch.exp(loaded_model(images))
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f"Accuracy of loaded model: {accuracy.item()*100}%")