# Image-Classification-Program
This repository contains programs that uses pre-trained keras models such as ResNet50V and uses it for image classification.

There are two files in this repository:-
1. Image_Classification_System.py :- This file contains the pre-trained model and trains the model on the dataset provided by the user. The model demands two datasets two train itself on. The training and validation datasets must be provided by the user for the results. The model saves itself in a file after the training and validation testing is done.
2. IMG_CL.py :- This file helps the user to classify an input image that is being provided. The file uses the already saved pre-trained model from the file named "Image_Classification_System.py" and helps the user to classify the input image. The user must provide the labels in this file to the program to correctly name the predicted result from the model. The labels provided by the user must same as the labels present in the dataset folder.
