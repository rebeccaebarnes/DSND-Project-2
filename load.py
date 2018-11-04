import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from collections import namedtuple
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Set data details
scale = 255 # Image reduction scale in pixels
input_shape = 224 # Required shape for nn
Data = namedtuple('Data', 'train valid test')
Details = namedtuple('Details', 'batch_size shuffle_setting')
loader_details = Data(Details(64, True), Details(32, False), Details(32, False))
mean = (0.485, 0.456, 0.406) # Color normalization mean
std = (0.229, 0.224, 0.225) # Color normalization std

# Load flower names
with open('cat_to_name.json', 'r') as f:
    flower_names = json.load(f)

def data_transforms():
    '''Creates train, valid(ation), and test data transform objects in named tuple'''
    data_transforms = Data(
        transforms.Compose([transforms.RandomResizedCrop(input_shape),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomRotation(degrees=90),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize(scale),
                            transforms.CenterCrop(input_shape),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize(scale),
                            transforms.CenterCrop(input_shape),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)]))
    
    return data_transforms

def image_datasets(data_dir, data_transforms):
    '''Transforms train, valid(ation), and test images according to data_transforms
       and returns results in named tuple'''
    # Create list
    image_datasets = [datasets.ImageFolder(os.path.join(data_dir, data_set),
                                           getattr(data_transforms, data_set)) \
                      for data_set in data_transforms._fields]
    # Convert to named tuple
    image_datasets = Data(image_datasets[0], image_datasets[1], image_datasets[2])

    return image_datasets

def data_loaders(image_datasets):
    '''Creates data loaders for train valid(ation), and test images and returns
       results in named tuple'''
    # Create list
    data_loaders = [torch.utils.data.DataLoader(image_dataset, 
                                                batch_size=details.batch_size,
                                                shuffle=details.shuffle_setting) \
                    for data_set, image_dataset, details \
                    in zip(image_datasets._fields, image_datasets, loader_details)] 
    # Convert to named tuple
    data_loaders = Data(data_loaders[0], data_loaders[1], data_loaders[2])

    return data_loaders

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: (int) size of the input
            output_size: (int) size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: (float) between 0 and 1, dropout probability
        '''
        super().__init__()

        # Add input layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([
            nn.Linear(h_input, h_output) for h_input, h_output in layer_sizes])
        
        # Add output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)

        # Include dropout
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        # Forward through each hidden layer with ReLU and dropout
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)

        # Pass through output layer
        x = self.output(x)

        return F.log_softmax(x, dim=1) # Using log to reduce impact of small decimals

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open image and get dimensions
    width, height = image.size
    
    # Change the longest side to 256 while maintaining aspect ratio
    # Get new size
    if width < height:
        divisor = width/256
        new_size = (256, int(height/divisor))
    else:
        divisor = height/256
        new_size = (int(width/divisor), 256)
    # Resize image
    image.thumbnail(new_size)
    
    # Center crop the image
    crop_size = 224
    left = round((new_size[0] - crop_size)/2)
    right = round((new_size[0] + crop_size)/2)
    top = round((new_size[1] - crop_size)/2)
    bottom = round((new_size[1] + crop_size)/2)
    cropped_image = image.crop((left, top, right, bottom)) # TODO: Make this work!!
    
    # Convert values to range of 0 to 1
    np_image = np.array(cropped_image)
    np_image = np_image/255
    
    # Standardize values
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    std_im = (np_image - means)/stds
    
    # Make the color channel the first channel
    final_image = std_im.transpose(2, 0, 1)
    
    return final_image

def imshow(image, ax=None, title=None):
    '''Imshow for Tensor.'''
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    if title is not None:
        ax.set_title(title)
    
    return ax

def pretrained_model(checkpoint):
    '''Loads a prediction model from checkpoint file (.pth)'''
    # Create new model
    checkpoint = torch.load(checkpoint)
    model = getattr(models, checkpoint['arch'])(pretrained=True)

    # Free paramaters
    for param in model.parameters():
        param.requires_grad = False
        
    # Create new classifier 
    classifier = Network(input_size=25088, 
                         output_size=102, 
                         hidden_layers=[516, 256])
    
    # Replace vgg classifier with new classifier
    model.classifier = classifier
    
    # Load saved state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer
    model.optimizer = checkpoint['optimizer']
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load previous epochs
    model.epochs = checkpoint['epochs']
    
    # Load class_to_idx
    model.class_to_idx = checkpoint['class_to_idx']

    return model

if __name__ == "__main__":
    # Test flower_names
    for i, key in enumerate(flower_names.keys()):
        print(key, ':', flower_names[key])
        if i == 5:
            break

    data_dir = 'flowers'
    # Test creation of all input files
    data_transforms = data_transforms()
    image_datasets = image_datasets(data_dir, data_transforms)
    data_loaders = data_loaders(image_datasets)
    # If three loader objects print, setup is complete
    for data_loader in data_loaders:
        print(data_loader)

    # Test creation of Network class
    classifier = Network(input_size=25088, output_size=102, hidden_layers=[516, 256])
    print(classifier)

    # Test image processing
    image_folder = 'flowers/train/'
    random_folder = random.choice(os.listdir(image_folder))
    random_file = random.choice(os.listdir(os.path.join(image_folder, random_folder)))
    image_path = os.path.join(image_folder, random_folder, random_file)
    image = Image.open(image_path)
    processed_image = process_image(image)
    display_image = imshow(processed_image)
    plt.show()
