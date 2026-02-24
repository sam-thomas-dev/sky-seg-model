## Synopsis
The python project in the "sky_seg" directory defines a convolutuion neural network used predict which pixels in an image are sky. The model makes this prediction by producing a Mask over an input image, which dictates which pixels the model predicts are sky.

### Motivation
My goal with this project was to teach myself how machine learning works at a fundemental level, using one the most popular deep learning tensor packages, that being PyTorch. Due to this goal, I didn't want to use a pre-defined model, I wanted to create my own model, for my own specific use case. And a creating a CV model for sky segmentation seemed like a good task for this purpose.  



## Document Content
This document outlines how the neural network was defined, what dataset was used, how to run the pre-trained model, and how run the model training loop. It contains the following sections:
- [Importing The Dataset](#importing-the-dataset)
- [Showing Data & Predictions](#showing-data--predictions)
- [Defining The Network](#defining-the-network)
- [Training & Testing](#training--testing)
- [How To Run The Model](#how-to-run-the-model)

## Importing The Dataset
The dataset chosen to train the model was the ["CityScapes Dataset"](https://www.cityscapes-dataset.com/), a popular choice for training segmentation models.

From this dataset, I used specifically the "gtFine_trainvaltest.zip (241mb)" and "leftImg8Bit_trainvaltest.zip (11GB)" datasets. 

leftImg8Bit contains a set of 8 bit images, and gtFine contains a set of labels/masks which correspond to each image, where each pixel in said mask is given an ID based on what it coresponds to in the image. For example, all pixels that represent sky in the image are given the value 23.

As the name suggests, this data comes pre-segemented into a training, validation, and testing set.

### Making The Data Useable
To use this dataset to train the model, we need use pytoch's ["Dataset"](https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html) class to create a subclass that will be used to load the data. This subclass defines methods that will be performed on the input data when the class is instantiated, this allows us to get the data into a form that the model can train on.

In this case, when the CityscapeSkyDataset class is instantiated and its get method is called by the Pytorch ["DataLoader"](https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html) (see below), the following actions are performed. An image and label mask are retrived from the appropriate dataset (as in validation, testing, etc..). 

The image is resized to match the model input size, converted to a tensor, and then its values are normalised which helps standardise feature between images. The label mask is then used to create a binary mask where each pixel with the value (ID) of 23 is assigened a value of 1 (with all other pixels being assigned 0), this create a mask with the same dimentions as the original image that outlines sky, this mask is then resized to the same dimentions as the resized image. With the manipulated image an mask then being returned as the output.

### Loading The Data

To load the data, all you have to do is create an instance of the CityscapeSkyDataset class, providing the respective parameters depending on whether you want it to represent the training, validation, or testing data. And create a new DataLoader instance using pytorch's ["DataLoader"](https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html) class.

The dataloader is used to define how the data is loaded. It takes 3 main parameters, dataset instance, batch size, and shuffle. Data set instance is the dataset subclass you wish to load data from, batch size is how many datapoint you would like to load on each iteration, and shuffle is used to specifiy whether you would like the order of the data to be shuffled. 

In this case, the training data is set to be loaded from the "trainingData" instance, in batches of 4, with the order shuffled.


## Showing Data & Predictions

## Defining The Network

## Training & Testing

## How To Run The Model










