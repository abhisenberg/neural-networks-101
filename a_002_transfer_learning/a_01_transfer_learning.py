"""
We will use pre-trained models and modify their last layers.
"""

import time
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict

# this is how you access the pretrained models, they'll be within some library and we simply access them using the dot notation
model = models.densenet121(pretrained = True)

# let's define the training and testing dataset
training_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# let's now load the images dataset
data_dir = "../training-data/pet_images"
training_dataset = datasets.ImageFolder(data_dir+"/train", transform=training_transform)
testing_dataset = datasets.ImageFolder(data_dir+"/test", transform=test_transform)

print(f"class to index mapping: {training_dataset.class_to_idx}")

trainloader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testing_dataset, batch_size=64)

# now let us modify the last classification layer to replace it with our own new classification layer
### first let us turn freeze the existing weights so that the pre-existing training will stay as-is
for param in model.parameters():
    param.requires_grad = False

### creating our own classification layer
cl = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 50)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(50, 2)),
    ('output', nn.LogSoftmax(dim=1))
]))
model.classifier = cl

# we will now test the model training performance on cpu vs gpu
for device in ['cpu', 'cuda']:

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001) #IMPORTANT: Only train the classification layer parameters

    model.to(device)

    for index, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        start = time.time()
        
        optimizer.zero_grad()
        output = model.forward(images)

        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        if index == 3:
            break
    
    print(f"Device={device}, time taken={(time.time() - start)/3:.3f} seconds")

