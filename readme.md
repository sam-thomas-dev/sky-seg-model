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


## Defining The Network
A neural network in pytoch is deffined by creating a subclass of pytorch's ["Torch.nn.Module"](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) class. This subclass is where you define the tensor oporations to be used in both forwad and backward propogation, along with how those operation are used to create the layers of the model. When the neural network is defined, you only need to define the forward propogation method, as the backward propogation will be defined automatically using ["autograd"](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) (Pytorch's automatic automatic differentiation engine, used to compute tensor gradients) based off the forward prop function.

### What Was Implemented
In this case, I have decided to define the model using the Mini U-Net achitecture, which as the name suggests is a more efficient scalled down version of the U-Net achitechture, whith less layers and starting at a smaller resolution, its a good choice for simple segmentation models. 

More specifically, the following is the exact architecture that was use for the network. 


Input (3, H, W) -> 
3x3 Conv2D -> 
ReLU -> 
3x3 Conv2D -> 
ReLU -> 
Downsample ->
3x3 Conv2D-> 
ReLu -> 
3x3 Conv2D-> 
ReLU -> 
Downsample -> 
Bottleneck -> 
Upsample -> 
3x3 Conv2D -> 
ReLU -> 
3x3 Conv2D -> 
ReLU ->
Upsample -> 
3x3 Conv2D -> 
ReLU -> 
3x3 Conv2D -> 
ReLU ->
1x1 Conv2D -> 
Output (1, H, W).
 
## Training & Testing
To train a model using pytorch, you need 4 things, an instance of your model, a data loader to access the training dataset, a loss function, and an optimiser. The two that have not been covered above are, the loss function, and the optimiser. 

### Loss Fuction
A loss function is used to quntify the difference between the models expected output, and its actual output. Its output is used in backware propogation to determine how the biasis and weights in the network should be adjusted. In this instance pytorch's ["BCEWithLogitsLoss"](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) was used. When the "backward()" is called on the loss object, the gradients are calcualted for each parameter. Once this is done, the optimiser can then be used to optimise said parameters.

### Optimiser
An optimiser is used to update a models parameter weights and biasis based on the gradients calculated by the loss function, in order to minimise loss. When an optimiser is instantiated, you must provide a learning rate, this specifies geat each step will be from the optimiser. In this case pytorchs ["torch.optim.Adam"](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html) optimiser has been used. After the loss function has computer the gradients for each parameter, the optimisers "step()" method is called, this adjusts each parameters weighs based on he gradients stored in said parameter. After this method has been run, the gradients stored for each parameter are zeroed using the optimisers "zero_grad()" method, if this is not done the gradients will accumulate each time the loss function calculates gradients, leading to incorrect parameter weight adjustments by the optimiser.


### Training & Testing Loop
Each time a model has been trained on an entire dataset, it is refered to as an epoch. In this case the model will be trained for 5 epochs. On each epoch, the model will be trained on all data in the dataset, after which a test function will caclulate the performace of the model using all data in the validation data set. 

This is done by takeing the logit tensor output by the model, and creating a mask using a threshold value, where every value above the threshold becomes 1, and all below become 0. In this case the threshold has been set to 0.5. Using this mask, both an IoU and Dice score are calculated for each image by comparing it with the actual sky mask for that image. Where the IoU & Dice score are two methods of quantifying the overlap of two masks, the higher the score the closer they are to being the same. After these scores have been calculated for each image, the average Dice & IoU score for all images will be displayed.

Once all epochs have been passed, the final weights and baiasis for each parameter in the model will be saved to a file labeled "model_params.pt".

## How To Run The Model
All required packages to run the model are specified in ["requirements.txt"](./requirements.txt), all of which can be downloaded using the `pip install -r requirements.txt` command. To run the model you will either need to download the ["CityScapes Dataset"](https://www.cityscapes-dataset.com), more detials on this can be found [here](./dataset_info.txt). Running main.py currently will load a test prediction from the validation data set using parameter weights and biasis I generated from 5 iterations of the training loop. To change this, simply call the other function in `main.py`, as specified below.

`main.py` contains two functions `runTrainingLoop()` and `loadPreTrainedPred()`. As their names suggest calling `runTrainingLoop()` will start a new instance of training the model, and calling `loadPreTrainedPred()` will load a prediction based on the pretrained weights.









