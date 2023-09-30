import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import sys
import os
from time import time

class DNN(nn.Module):
   
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size[1], output_size)
        self.logsm = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.logsm(x)
        return x 
    
    def train(self, model, device, train_loader, learning_rate, num_epochs):
        
        # Train the model
        loss_fn = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_loss = 0
        correct = 0
        accuracies = []
        
        print("Start Training ...")

        for epoch in range(num_epochs):

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                n_data = data.view(-1, data.size()[2] * data.size()[3])
                output = model(n_data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

            accuracy = 100. * correct / len(train_loader.dataset)
            accuracies.append(accuracy)
            
            print('Train Epoch {}:: Accuracy: {}/{} ({:.0f}%)'.format(
                epoch+1, correct, len(train_loader.dataset), accuracy))
            correct = 0
            
        print("Finished Training")
        
        # save model checkpoint
        print("Saving the model")
        torch.save(model.cpu().state_dict(), os.path.join("../Homework1", "model.pth"))
        print("Saved the model")

        return accuracies

    def test(self, model, device, test_loader, num_epochs):
        
        # Test the model
        loss_fn = nn.NLLLoss()
        
        test_loss = 0
        correct = 0
        accuracies = []
        
        print("Start Testing ...")
        
        for epoch in range(num_epochs):
        
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)

                    n_data = data.view(-1, data.size()[2] * data.size()[3])
                    output = model(n_data)
                    test_loss += loss_fn(output, target)
                    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

                accuracy = 100. * correct / len(test_loader.dataset)
                accuracies.append(accuracy)
                
                print('Test Epoch {}:: Accuracy: {}/{} ({:.0f}%)'.format(
                    epoch+1, correct, len(test_loader.dataset), accuracy))
                correct = 0
                
        print("Finished Testing")
        
        return accuracies