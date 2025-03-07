import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self, input_nodes, output_nodes, hidden_nodes_list, drop_p=0.5):        
        super.__init__()

        # create the input layer
        self.input_layer = nn.Linear(input_nodes, hidden_nodes_list[0])

        # create a list of tuples for the hidden layer nodes, and create hidden layers
        hidden_layer_sizes = zip(hidden_nodes_list[:-1], hidden_nodes_list[1:])
        self.hidden_layers = nn.ModuleList([nn.input(h1, h2) for (h1, h2) in hidden_layer_sizes])

        # create the output layer
        self.output_layer = nn.Linear(hidden_nodes_list[-1], output_nodes)
        
        # create the dropout function wrapper
        self.dropout = nn.Dropout(p=drop_p)
    
    def forward(self, x):

        # result from input layer
        x = self.dropout(F.relu(self.input_layer(x)))

        # result from other hidden layers
        for hidden in self.hidden_layers:
            x = F.relu(hidden)
            x = self.dropout(x)
        
        # result from final output
        x = F.log_softmax(self.output_layer(x))
        return x
    
def train(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every=50):
    steps = 0

    for e in range(epochs):
        runningloss = 0
        for images, labels in trainloader:
            steps += 1
            images = images.view(images.shape[0], -1)
            
            optimizer.zero_grad()   # clear the previous grads

            output = model.forward(images)   # get the output 
            loss = criterion(output, labels)    # get the loss
            loss.backward()     #calculate the grads
            optimizer.step()    #do the GD

            runningloss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
                running_loss = 0

                model.train()

def validation(model, testloader, criterion):
    accuracy, loss = 0, 0
    for images, labels in testloader:
        images = images.view(images.shape[0], -1)
        output = model.forward(images)
        
        # calculate the loss
        loss += criterion(output, labels)

        # calculate the accuracy
        ps = torch.exp(output)  #since output is in log_softmax, take exp to get actual probs
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
    return loss, accuracy