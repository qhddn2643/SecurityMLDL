import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import sys
import os
import matplotlib.pyplot as plt
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
        torch.save(model.cpu().state_dict(), os.path.join("../Homework2", "model.pth"))
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
    
    
    # FGSM attack code
    def fgsm_attack(self, image, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image

    def adv_test(self, model, device, test_loader, epsilon):
        
        # Accuracy counter
        correct = 0
        adv_examples = []
        final_accuracies = []
        
        # Test the model
        print("Start Adversarial Examples Testing ...")
                
        # Loop over all examples in test set
        for data, target in test_loader:

            # Send the data and label to the device
            data, target = data.to(device), target.to(device)

            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True
            n_data = data.view(-1, data.size()[2] * data.size()[3])

            # Forward pass the data through the model
            output = model(n_data)
            init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            # If the initial prediction is wrong, dont bother attacking, just move on
            if init_pred.item() != target.item():
                continue

            # Calculate the loss
            loss = F.nll_loss(output, target)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = self.fgsm_attack(data, epsilon, torch.sign(data_grad))
            n_pertu_data = perturbed_data.view(-1, perturbed_data.size()[2] * perturbed_data.size()[3])

            # Re-classify the perturbed image
            output = model(n_pertu_data)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            if final_pred.item() == target.item():
                correct += 1
                # Special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 10):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((adv_ex, init_pred.item(), final_pred.item()))
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 10:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((adv_ex, init_pred.item(), final_pred.item()))

        # Calculate final accuracy for this epsilon
        final_acc = 100. * correct / len(test_loader.dataset)
        final_accuracies.append(final_acc)

        print("Epsilon: {}\tAccuracy: {}/{} ({:.0f}%)".format(
            epsilon, correct, len(test_loader.dataset), final_acc))
        correct = 0

        print("Finished Adversarial Examples Testing\n")
        
        return final_accuracies, adv_examples