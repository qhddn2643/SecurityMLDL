import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision
from tqdm import tqdm
from typing import List

    
class ResNet(nn.Module):
    
    def train(model, device, train_loader, learning_rate, num_epochs, is_defensed):
        
        # Train the model
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_loss = 0
        correct = 0
        accuracies = []
        predicts = []
        
        if is_defensed == True:
            train_loader.dataset
        
        print("Start Training ...")

        for epoch in range(num_epochs):

            for batch_idx, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                output = model(inputs)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()

                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                
                if is_defensed == True:
                    # Apply Majority Voting
                    best_pred = Defense.MajorityVoting(pred)
                    correct += best_pred.eq(labels.view_as(best_pred)).sum().item()
                else:
                    correct += pred.eq(labels.view_as(pred)).sum().item()
                
            accuracy = 100. * correct / len(train_loader.dataset)
            accuracies.append(accuracy)
            
            print('Train Epoch {} | Accuracy: {}/{} ({:.0f}%)'.format(
                epoch+1, correct, len(train_loader.dataset), accuracy))
            correct = 0
            
        print("Finished Training")

        return accuracies
    
    
    def test(model, device, test_loader, num_epochs, is_defensed):
        
        # Test the model
        
        test_loss = 0
        correct = 0
        accuracies = []
        predicts = []
        
        if is_defensed == True:
            test_loader.dataset
        
        print("Start Testing ...")
        
        for epoch in range(num_epochs):
        
            with torch.no_grad():
                for batch_idx, (inputs, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
                    inputs, labels = inputs.to(device), labels.to(device)

                    output = model(inputs)
                    test_loss += F.cross_entropy(output, labels)
                    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                    
                    
                    if is_defensed == True:                    
                        # Apply Majority Voting
                        best_pred = Defense.MajorityVoting(pred)
                        correct += best_pred.eq(labels.view_as(best_pred)).sum().item()
                        
                    else:
                        correct += pred.eq(labels.view_as(pred)).sum().item()
                
                accuracy = 100. * correct / len(test_loader.dataset)
                accuracies.append(accuracy)
                
                print('Test Epoch {} | Accuracy: {}/{} ({:.0f}%)'.format(
                    epoch+1, correct, len(test_loader.dataset), accuracy))
                correct = 0
                
        print("Finished Testing")
                
        return accuracies
    
       
class Attack(nn.Module): # Model Poisoning Attack
    
    def poison_image(self, image): # Poison the images
        """apply watermark to image"""
        image[0,-1,-1]=1.
        image[0,-1,-3]=1.
        image[0,-3,-1]=1.
        image[0,-3,-3]=1.
        image[0,-2,-2]=1.
        return image

    def __init__(self, dataset, poison_probability, poison_label):
        # apply poison
        self.dataset = dataset
        self.poison_probability = poison_probability
        self.poison_label = poison_label
        
    def DataPoisoning(self):
        new_dataset = [(self.poison_image(images), self.poison_label) if np.random.rand() < self.poison_probability else (images, labels) for images, labels in self.dataset]
        return new_dataset

    

class Defense(nn.Module):
    
    def NewDifferentModels(models):
        
        model1 = models.resnet18(pretrained=True)
        model2 = models.resnet34(pretrained=True)
        model3 = models.resnet50(pretrained=True)
        
        model_list = [model1, model2, model3]

        return model_list
    
    def MajorityVoting(predicts):
        result = []
        for i in range(len(predicts)):
            t = torch.tensor(predicts[i])
            result.append(torch.mode(t, dim=0)[0])
        return torch.tensor(result)
    
    
class Randomization(nn.Module):
    
    def randomPadding(self, images):
        
        pad = 5
        rnd_generate_num = 0
        random_genereated_images = []
        
        rnd = random.randint(len(images[0]), len(images[0])+5)

        transform_img = torchvision.transforms.Resize((rnd, rnd))
        rezised_images = transform_img(images)

        for i in range(rnd):
            random_genereated_images.append(rezised_images)

        random_images = torch.stack(random_genereated_images)
        
        padded_images = F.pad(random_images, (rnd, rnd), "constant", 0) # random zero-padding
        
        pad_w = len(padded_images) - len(random_images) + 1
        pad_h = len(padded_images) - len(random_images) + 1
        
        new_transform_img = torchvision.transforms.Resize((pad_w, pad_h))
        new_padded_images = new_transform_img(padded_images)

        select_one_image = random.choice(new_padded_images)
        return select_one_image
    

    def __init__(self, original_dataset):
        self.dataset = [(self.randomPadding(images), labels) for images, labels in original_dataset]
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx] 
