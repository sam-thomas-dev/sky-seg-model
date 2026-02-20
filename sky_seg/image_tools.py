# From dependencies
import torch
import matplotlib.pyplot as plt

# From external modules
import image_tools

def unnormaliseImage(img):
    """
    Used to unnormalise a normalised image stored in a tensor
    
    Args:
        img (tensor): Normalised tensor representing an image 

    Returns:
        tensor: Tensor with unnormalised values representing an image
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return img * std + mean

def showXImagePreds(images, masks, preds, batch_size):
    """
    Used to show image, image mask, and predicted mask from model.
    Note, the number of images shown can not be more than the batch size.
    
    :param images: Iterator containing all images for batch
    :param masks: Iterator containing all masks for batch
    :param preds: Iterator containing all mask predictions for batch
    :param batch_size: The number of images in batch
    """
    _, axes = plt.subplots(4, 3, figsize=(12, 12))
    axes[0, 0].set_title("Image")
    axes[0, 1].set_title("Mask")
    axes[0, 2].set_title("Pred Mask")

    for i in range(batch_size):
        img = image_tools.unnormaliseImage(images[i].cpu()).permute(1, 2, 0)
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

def showInputData(images, masks, numToShow):
    """
    Used to show just the training or prediction images and masks that will be used as inputs for the model
    
    :param images: Iterator containing all images in batch
    :param masks: Iterator containing all masks in batch
    :param numToShow: Number of images out of batch to show, can not be higher than batch size
    """
    plt.figure(figsize=(12,6))

    for i in range(numToShow):
        img = image_tools.unnormaliseImage(images[i].cpu()).permute(1, 2, 0)
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