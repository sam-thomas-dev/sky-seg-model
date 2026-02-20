# From dependencies
import torch

# From extrnal modules
import image_tools


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

        if batch % 50 == 0:
            print("Train Progress:") 
            print(f"Data Passed: {batch*batch_size}/{size}")
            print(f"Batch Loss: {loss.item()}")
            

def iou_calc(pred_mask, mask):
    """
    Calcualtes IoU score of two boolean tensors (both tensors must only contain 1 or 0).\n
    Calculates how many of the total joined 1 values are in the same position.\n
    Value returned is 1 if both tensors have all 1s in the same positions  
    Args:
        pred_mask: tensor representing prediction mask
        mask: tensor representing correct mask
    """
    # pred n mask / pred + mask - (pred n mask)
    pred_intersection = torch.logical_and(mask, pred_mask)
    pred_union = torch.logical_or(mask, pred_mask)
    return pred_intersection.sum() / pred_union.sum()

def dice_calc(pred_mask, mask):
    """
    Calcualtes Dice score of two boolean tensors (both tensors must only contain 1 or 0).\n
    Calculates how many of the total joined 1 values are in the same position (using different method than IoU).\n
    Value returned is 1 if both tensors have all 1s in the same positions  
    Args:
        pred_mask: tensor representing prediction mask
        mask: tensor representing correct mask
    """
    # (pred n mask)*2 / pred + mask 
    pred_intersection = torch.logical_and(mask, pred_mask)
    pred_union = torch.logical_or(mask, pred_mask)
    return (pred_intersection.sum()*2) / (pred_union.sum() + pred_intersection.sum())



def test_loop(dataloader, model_inst, batch_size):
    # Set the model to evaluation mode, 
    # important for batch normalization and dropout layers
    model_inst.eval()

    iou_total = 0.0
    dice_total = 0.0
    n = 0.0

    # Evaluating the model with torch.no_grad() 
    # ensures that no gradients are computed during test mode
    with torch.no_grad():
        for images, masks in dataloader:
            
            logits = model_inst(images)
            probs = torch.sigmoid(logits)

            #determines probability threshold for pixels that will make up the mask
            preds = (probs > 0.5).float()

            for i in range(batch_size):
                iou_total += iou_calc(preds[i], masks[i])
                dice_total += dice_calc(preds[i], masks[i])
                n+=1

        # show avg dice & iou score of all images in validation set
        print("Test Results:")          
        print(f"IoU:  {iou_total / n}")
        print(f"Dice: {dice_total / n}")
            

def loadTestPrediction(valDataLoader, model_inst, batch_size):
    """
    Loads a prediction test that you can see, allowing you to compare masks and inital image

    Args:
        valDataLoader: Dataloader for validation data set
        model_inst: Model instance you want to make prediction with
        batch_size: Number of images in are iteration of data loader 
    """
    # Set the model to evaluation mode, 
    # important for batch normalization and dropout layers
    model_inst.eval()

    # Evaluating the model with torch.no_grad() 
    # ensures that no gradients are computed during test mode
    with torch.no_grad():
            images, masks = next(iter(valDataLoader))
            
            logits = model_inst(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            image_tools.showXImagePreds(images, masks, preds, batch_size)

