# Imports here
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms, models
#from torchvision.datasets import ImageFolder
import torchvision.models as models
import torchvision.transforms as transforms

from collections import OrderedDict
from PIL import Image
from os import listdir
import json
import argparse

#create argument parser object
parser = argparse.ArgumentParser()

#add arguments into the parser, per GPT
parser.add_argument('data_dir', type=str, help='path to the data directory')
parser.add_argument('--save_dir', type=str, help='directory to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg16', help='architecture of the model')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--hidden_units', type=int, default=2048, help='number of hidden units')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--gpu', action='store_true', help='use GPU for training')

#returns an object containing the values of the arguments
args = parser.parse_args()
#print for test
print(args.data_dir)
print(args.save_dir)
print(args.arch)
print(args.learning_rate)
print(args.hidden_units)
print(args.epochs)
print(args.gpu)

#assigns argparse inputs to variables to be used in the code below
data_dir = args.data_dir #dont know if i need
save_dir = args.save_dir #dont know if i need
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu



#how to use command line
#python train.py data_directory --save_dir save_directory --arch vgg13 --learning_rate 0.01 --hidden_units 512 --epochs 20 --gpu


#start copy paste from part 1 

#dont need below because data dir is argsparse
#data_dir = 'flowers'
train_dir = data_dir + '/train' #dont know why i had this commented out
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# For the training, transformations such as random scaling,
#cropping, and flipping. input data is resized to 224x224 pixel
train_transforms = transforms.Compose([
        transforms.RandomRotation(45), #random rotation
        transforms.RandomResizedCrop(224), #random cropping and resizing
        transforms.RandomHorizontalFlip(), #random flipping
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #normalize means and stdevs
    ])

#The validation and testing...For this you don't want any 
#scaling or rotation transformations, but you'll need to 
#resize then crop the images to the appropriate size.

valid_transforms = transforms.Compose([
        transforms.Resize(256), #resizes to 256, so all images same size
        transforms.CenterCrop(224), #then crops to center 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
     ])

# TODO: Load the datasets with ImageFolder
# reference format from  part 4 MNIST example
train_dataset = datasets.ImageFolder(data_dir, transform = train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)


# TODO: Using the image datasets and the trainforms, define the dataloaders

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64, shuffle = True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = True)



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
#Build and train your network    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
#Load a [pre-trained network] vgg16
#old attempt
#model = models.arch(pretrained = True) #PRETRAINED IS OBSOLETE, arch is argeparse architecture

model = models.__dict__[arch](pretrained=True) #from GPT, did not try

for param in model.parameters():
    param.requires_grad = False
    
#DEFINE NETWORK ARCHITECTURE
#FEED FORWARD

classifier = nn.Sequential(OrderedDict([ 
                          ('fc1',nn.Linear(25088,2048)),
                           ('ReLu1',nn.ReLU()),
                           ('Dropout1',nn.Dropout(p=0.5)),
                           ('fc2',nn.Linear(2048,hidden_units)), #try 2048, 2048
                           ('ReLu2',nn.ReLU()),
                           ('Dropout2',nn.Dropout(p=0.5)),
                           ('fc3',nn.Linear(hidden_units,102)), #try 2048, 102
                           ('output',nn.LogSoftmax(dim=1))
]))
model.classifier = classifier
      
#define loss and optimizer 
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)   

model.to(device);

#epochs = 3 use argsparse value
steps = 0
running_loss = 0
print_every = 10

for epoch in range(epochs):
    running_loss = 0
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() #end training loop
        
        #begin validation loop        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader: #migth need to be validloader
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0 #not sure why this is here
            model.train()
            
# TODO: Save the checkpoint 

#think this next line is redundant
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
class_to_idx = train_data.class_to_idx
model.class_to_idx = class_to_idx

#image_datasets = datasets.ImageFolder(data_dir, transform=train_transforms)

checkpoint = { #these are the features that should be saved and loaded, note that in the predict.py we will need to load these same feature
    'classifier':model.classifier,
    'model_state_dict': model.state_dict(),
    'epochs': epochs,
    'optimizer_state_dict': optimizer.state_dict(),
    'class_to_idx': class_to_idx # was image_datasets.class_to_idx
}
torch.save(checkpoint, 'checkpoint.pth') #saves the above features to checkpoint.pth
print("Model Saved")