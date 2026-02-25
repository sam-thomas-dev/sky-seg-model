# From dependencies
import torch.nn as nn
import torch

# From extrnal modules
import nn_definition
import dataset_tools
import model_training_testing

# ---Training Model & Saving Learned Parameters---
def runTrainingLoop():
    learning_rate = 1e-3
    batch_size = 4
    epochs = 5

    modelInstance = nn_definition.SegmentationModel()
    lossFunction = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(modelInstance.parameters(), lr=learning_rate)

    # trains model on entire dataset X times
    for iter in range(epochs):
        print(f"Epoch: {iter}\n")
        model_training_testing.train_loop(dataset_tools.trainLoader, modelInstance, lossFunction, optimiser, batch_size)
        model_training_testing.test_loop(dataset_tools.valLoader, modelInstance, batch_size)
        print(f"----------\n")

    # saves only model weights & parameters
    torch.save(modelInstance.state_dict(), "./sky_seg/model_params.pt")

# ---Loading A Prediction Using Trained Weights---
def loadPreTrainedPred():
    model_inst = nn_definition.SegmentationModel()
    model_inst.load_state_dict(torch.load("./model_params.pt", weights_only=True))
    model_inst.eval

    model_training_testing.loadTestPrediction(dataset_tools.valLoader, model_inst, 4)

loadPreTrainedPred()