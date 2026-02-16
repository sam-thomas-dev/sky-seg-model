import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

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

#Chains multiple torchvision transformation opperations togther into 1 operation 
transformImage = T.Compose([
    T.Resize((256, 512)), #resizes input image to specified size, where if image is tensor, its expected shape is [..., H, W], where ... is a max of 2 leading dimentions
    T.ToTensor(), #converts PIL image or np.array to tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #transforms tensor to have 0 mean & unit variance, helps standardise features and bring data into a consistant range
])

#Creates dataset instance taking images & labels from training subdirectory
trainingData = CityscapeSkyDataset(
    root="cityscape_data",
    split="train",
    transform=transformImage
)

#Creates dataset instance taking images & labels from validation subdirectory
validationData = CityscapeSkyDataset(
    root="cityscape_data",
    split="val",
    transform=transformImage
)

#Creates dataloader that loads 4 images at a time for both training and validation
trainLoader = DataLoader(trainingData, batch_size=4, shuffle=True, num_workers=0)
valLoader = DataLoader(validationData, batch_size=4, shuffle=False)

def unnormaliseImage(img):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        return img * std + mean

def showIterData(images, masks, numToShow):
    plt.figure(figsize=(12,6))

    for i in range(numToShow):
        img = unnormaliseImage(images[i].cpu()).permute(1, 2, 0)
        msk = masks[i].cpu().squeeze()

        plt.subplot(2, numToShow, i+1)
        plt.title(f"img{i}")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(2, numToShow, i+numToShow+1)
        plt.title(f"msk{i}")
        plt.imshow(msk, cmap="Blues")
        plt.axis("off")

    plt.show()

#setup model
# mini u-net architecture, 
# input (3, H, W) -> ((conv->relu)x2 -> downsample(maxpool))x2 -> bottleneck -> (upsample -> (conv->relu)x2)x2 -> 1x1 conv -> (1, H, W) 

# class DoubleConvolution(nn.Module):
#     def __init__(self, input_ch, output_ch):
#         super().__init__()
#         #runs the following tensor operations sequentially
#         self.net = nn.Sequential(
#             nn.Conv2d(input_ch, output_ch, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(output_ch, output_ch, 3, padding=1),
#             nn.ReLU(inplace=True)
#         )
    
#     def forward(self, x):
#         return self.net(x)

def dblConv(channel_in, channel_out):
    '''
    Creates a tensor opperation that runs a convolution layer and activation function twice, 
    turing the number of input channels specified into the number of output channels specified
    
    :param channel_in: number of channels of input image tensor
    :param channel_out: desired number of output channels
    '''
    return nn.Sequential(
        nn.Conv2d(channel_in, channel_out, 3, padding=1), #definies a conv layer/funcion that turns x input channels into x feature maps, using a 3x3 kernel, adding a 1 pixel boarder/padding of 0s around the image
        nn.ReLU(inplace=True), #calls ReLU activation function, overwrites input tenser with its output to save memory (thats what inplace does)
        nn.Conv2d(channel_out, channel_out, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class SegmentationModel(nn.Module):

    def __init__(self):
        super(SegmentationModel, self).__init__()
        # Encoder
        self.conv1 = dblConv(3, 64) #runs dblConv, turning 3 input channels into 64 feature maps

        self.conv2 = dblConv(64, 128) #runs dblConv, turning 64 input channels into 128 feature maps
        
        self.maxpool = nn.MaxPool2d(2) #reduces the image tensors size by a factor of 2 (takes the largest value from each 2x2 section of the image tensor)

        # Bottleneck
        self.bottleneck = dblConv(128, 256) #runs dblConv, turning 128 input channels into 256 feature maps

        # Decoder
        self.upsample1 = nn.ConvTranspose2d(256, 128, 2, stride=2) #upconvolution upsamples 2x (as specified in stride) using a 2x2 kernel, going from 256 to 128 channels
        self.conv3 = dblConv(256, 128) #runs dblConv, turning 256 input channels into 128 feature maps

        self.upsample2 = nn.ConvTranspose2d(128, 64, 2, stride=2) #upconvolution upsamples 2x (as specified in stride) using a 2x2 kernel, going from 256 to 128 channels
        self.conv4 = dblConv(128, 64) #runs dblConv, turning 128 input channels into 64 feature maps

        # Output head
        self.outputConv = nn.Conv2d(64, 1, kernel_size=1) #definies a conv layer/funcion that turns 64 input channels into 1 feature map, using a 1x1 kernel

    def forward(self, x):
        # Encoder
        e1 = self.conv1(x)
        e2 = self.conv2(self.maxpool(e1))

        # Bottleneck
        b = self.bottleneck(self.maxpool(e2))

        # Decoder
        d2 = self.upsample1(b)
        d2 = torch.cat([d2, e2], dim=1) #concatenates tensors d2 & e2 in that order along the y axis
        d2 = self.conv3(d2)

        d1 = self.upsample2(d2)
        d1 = torch.cat([d1, e1], dim=1) #concatenates tensors d1 & e1 in that order along the y axis
        d1 = self.conv4(d1)

        # logits
        return self.outputConv(d1)
    
#training
# recomended for training, refer to pytorch documentation for how to create training and testing loops
# logits = modelInstance(images)
# loss = lossFunction(logits, masks)
# probs = torch.sigmoid(logits)
# preds = (probs > 0.5).float() #determines probability threshold for whats consiered sky

learning_rate = 1e-3
batch_size = 4
epochs = 5

modelInstance = SegmentationModel()
lossFunction = nn.BCEWithLogitsLoss()
optimiser = torch.optim.Adam(modelInstance.parameters(), lr=learning_rate)

def train_loop(dataloader, model_inst, loss_fn, optimizer, batch_size):
    # Set the model to training mode, important for batch normalization and dropout layers
    model_inst.train()
    
    size = len(dataloader.dataset)

    for batch, (images, masks) in enumerate(dataloader):
        # Compute prediction and loss
        logits = model_inst(images)
        loss = loss_fn(logits, masks)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# they are both just formulas, you can implement it yourself :)
# def iou_score(pred, mask, eps=1e-6):
#                 #also need to handle for if pred.sum() + target.sum() are 0
#                 intersection = (pred * mask).sum()
#                 union = pred.sum() + mask.sum() - intersection
#                 return (intersection + eps) / (union + eps)

# def dice_score(pred, mask, eps=1e-6):
#     intersection = (pred * mask).sum()
#     return (2 * intersection + eps) / (pred.sum() + mask.sum() + eps)

def test_loop(dataloader, model_inst, batch_size):
    # Set the model to evaluation mode, important for batch normalization and dropout layers
    model_inst.eval()

    iou_total = 0
    dice_total = 0
    n = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():
        for images, masks in dataloader:
            
            logits = model_inst(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float() #determines probability threshold for whats consiered sky

            for i in range(batch_size):
                #figure out way to calculate IoU score
                #figure out way to calculate dice coefficient
                n+=1
            
            #prints at the end of each batch
            print(f"IoU:  {iou_total / n:.4f}")
            print(f"Dice: {dice_total / n:.4f}")
            

def loadTestPrediction(valDataLoader, model_inst, batch_size):
    # Set the model to evaluation mode, important for batch normalization and dropout layers
    model_inst.eval()

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():
            images, masks = next(iter(valDataLoader))
            _, axes = plt.subplots(4, 3, figsize=(12, 12))

            logits = model_inst(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            axes[0, 0].set_title("Image")
            axes[0, 1].set_title("Mask")
            axes[0, 2].set_title("Pred Mask")

            for i in range(batch_size):
                img = unnormaliseImage(images[i].cpu()).permute(1, 2, 0)
                msk = masks[i].cpu().squeeze()
                pred = preds[i].cpu().squeeze()
                
                axes[i, 0].imshow(img)
                axes[i, 0].axis("off")

                axes[i, 1].imshow(msk)
                axes[i, 1].axis("off")

                axes[i, 2].imshow(pred)
                axes[i, 2].axis("off")

            plt.tight_layout()
            plt.show()

test_loop(valLoader, modelInstance, batch_size)
