import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import torch

# Cityscapes labels work by assigning each pixel in an image an ID,
# each ID represents a category, 
# this creates a label image the same size as the initial image,
# where the value of each pixel in said image corresponds to a category ID,
# for example, all pixel designated as sky have the value 23

def getSkyMask(label_path):
    '''
    Gets a mask of all skypixels from the label image specified in argument

    Args:
        label_path: string representing path of label image

    Returns:
        (uint8)np.array: binary array to be used as mask where 1 = sky pixel
    '''
    #ID for pixels that are labeled as sky
    SKY_ID = 23
    
    #opens image at specified path representing labels for input image
    labelsAsImg = Image.open(label_path)

    #converts pixel values of image into np.array
    label = np.array(labelsAsImg)
    
    #creates a boolean np.array where every value in label is compared with the value in sky_id.
    #each pixel is then replaced with the boolean result of that operation (ie, True or False),
    boolArray = label == SKY_ID

    #Array is then converted into the datatype uint8, meaning True = 1 and False = 0,
    #this results in a binary array where pixels that are 1 have the pixel id specified,
    #and all with 0 dont, so in this case, all 1 pixels had the id for sky,
    # this results in a skymask that can be used for training
    skyMask = boolArray.astype(np.uint8)

    return skyMask

class CityscapeSkyDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        #assigns arguments specified at instantiation to object feilds
        self.root = root
        self.split = split #only valid arguments are "train", "val", and "test"
        self.transform = transform

        #assigns images & labels feild to empty arrays
        self.images = []
        self.labels = []

        #Takes root feild and joins it with cityscape specific dataset direcotry name and split feild,
        #where root = folder where the leftImg8bit & gtFine datasets are stored, 
        #and split = training or validation subdirectory directory of those datasets.
        #Results in path equal to either training or validation root directory for both images & lables, depending on value of split feild
        img_root = os.path.join(root, "leftImg8bit", split)
        lbl_root = os.path.join(root, "gtFine", split)

        #For each subdirectory in the image_root path 
        for subDir in os.listdir(img_root):

            #For current subdirectory, gets directory of images and labels
            img_dir = os.path.join(img_root, subDir)
            lbl_dir = os.path.join(lbl_root, subDir)

            #For each file in image directory of subdirectory
            for file in os.listdir(img_dir):
                #If current file is valid dataset image
                if file.endswith("_leftImg8bit.png"):
                    #Create path to current image and corresponding label file by joining appropriate directory and filename
                    img_path = os.path.join(img_dir, file)
                    #End portion of file name replaced with appropriate naming convention for label
                    lbl_path = os.path.join(lbl_dir, file.replace("_leftImg8bit.png", "_gtFine_labelIds.png")) 
                    
                    #Adds current file/image and its corresponding label to arrays of both the images and lablels feilds
                    self.images.append(img_path)
                    self.labels.append(lbl_path)
    
    def __len__(self):
        #returns number of items in images feild
        return len(self.images)


    def __getitem__(self, idx):
        #Selects path at index specfied from labels & images array feilds 
        imagePath = self.images[idx]
        labelPath = self.labels[idx]

        #Opens image at filepath selected, converts it to RGB 3 channel colour 
        image = Image.open(imagePath).convert("RGB")

        #Gets a mask of all pixels in label image with id for sky
        mask = getSkyMask(labelPath)

        #if transform function provided, transform image
        if self.transform:
            image = self.transform(image)

        #Converts mask np.array to tensor, adds new dimention with size of 1 at index 0 of tensor,
        #so tensors shape now equals (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  

        #returns image and mask tupple
        return image, mask
