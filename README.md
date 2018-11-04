# Image Classifier
This project was completed as part of the course requirements of Udacity's [Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) certification.

## Overview
The project used transfer learning to train pre-trained ImageNet neural networks to identify 102 flower types from [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). An exploration was conducted into a notebook and then converted to executable scripts with the use or argparse to provide execution options.

The final products provide the user with the ability to: 
1. Specify the architecture of a classifier (with relu activation and dropout = 0.5 for hidden layers)
2. Substitute this classifier for that of a pre-trained VGG or DenseNet neural network
3. Train the classifier, printing incremental train/validation loss and validation accuracy
4. Save and load an existing model
5. Conduct a random search of optimizer, learning rate and epochs hyperparameters
6. Choose an image for prediction or select random image(s)
7. Make top-k predictions of images
8. Display predictions in a visual format with or without actual flower names

## Technologies Used
- PyTorch
- Python
- Libraries: numpy, matplotlib.pyplot, json, os, random, collections, PIL, argparse
- Jupyter Notebooks

## Key Findings
- Adam optimizer was much faster at the initial training than RMSprop
- Adagrad optimizer was slower than both the above
- Models benefited from reduced learning rates as they neared higher levels of accuracy to continue learning
- The final model had a classifier with two hidden layers of [516, 256], Adam optimizer, initial learning rate of 0.0003089, and trained on 58 epochs with 88% accuracy for the validation set and 86% accuracy for the test set.
- Some images were correctly predicted with almost 100% confidence of their type for their top 1 prediction, whereas others could be below 40%
- For images incorrectly predicted in the top 1 prediction, most had the correct flower type in the top 5 predictions. Generally incorrect images had lower top 1 confidence percentages, but at times the confidence could be over 80%
- It appeared that images that contained a front on view of the flower had a higher chance of being classified correctly compared to images take from other angles
