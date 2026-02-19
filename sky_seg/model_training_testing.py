# From dependencies
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

# From extrnal modules
import dataset_tools
import image_tools
import nn_definition

#Chains multiple torchvision transformation opperations togther into 1 operation 
transformImage = T.Compose([
    T.Resize((256, 512)), #resizes input image to specified size, where if image is tensor, its expected shape is [..., H, W], where ... is a max of 2 leading dimentions
    T.ToTensor(), #converts PIL image or np.array to tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #transforms tensor to have 0 mean & unit variance, helps standardise features and bring data into a consistant range
])

# Training Data
#Creates dataset instance taking images & labels from training subdirectory
trainingData = dataset_tools.CityscapeSkyDataset(
    root="cityscape_data",
    split="train",
    transform=transformImage
)

#Creates dataset instance taking images & labels from validation subdirectory
validationData = dataset_tools.CityscapeSkyDataset(
    root="cityscape_data",
    split="val",
    transform=transformImage
)

#Creates dataloader that loads 4 images at a time for both training and validation
trainLoader = DataLoader(trainingData, batch_size=4, shuffle=True, num_workers=0)
valLoader = DataLoader(validationData, batch_size=4, shuffle=False)

#training
# recomended for training, refer to pytorch documentation for how to create training and testing loops
# logits = modelInstance(images)
# loss = lossFunction(logits, masks)
# probs = torch.sigmoid(logits)
# preds = (probs > 0.5).float() #determines probability threshold for whats consiered sky

learning_rate = 1e-3
batch_size = 4
epochs = 5

modelInstance = nn_definition.SegmentationModel()
lossFunction = nn.BCEWithLogitsLoss()
optimiser = torch.optim.Adam(modelInstance.parameters(), lr=learning_rate)

# Trains models based on output
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

# Tests model during training loop to see accuracy after fitting
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
            


# used for model to predict where sky is on images never seen before
def loadTestPrediction(valDataLoader, model_inst, batch_size):
    # Set the model to evaluation mode, important for batch normalization and dropout layers
    model_inst.eval()

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():
            images, masks = next(iter(valDataLoader))
            
            logits = model_inst(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            image_tools.showXImagePreds(images, masks, preds, batch_size)

loadTestPrediction(valLoader, modelInstance, batch_size)
