import torch

# imports pretrained model and weights
from torchvision.models import resnet18, ResNet18_Weights

# calls pretrained model with weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# creates a random tensor to represent 1 image, with 3 channels, and a height & width of 64
data = torch.rand(1, 3, 64, 64)

# creates corresponding model labels with random values, label in pretrained model has the shape (1, 1000)
labels = torch.rand(1, 1000)

# Runs the input data through the model, 
# through each model layer to make a prediction, 
# this is the forward pass or forward propogation step
prediction = model(data)

# The models prediction and corresponding label is then used to calculate the error (or loss).
loss = (prediction - labels).sum()

# The error (loss) is then backward propogated (backward passed) through the network.
# Backward propogation is started when the '.backward()' method is called on the error tensor, 
# where autograd then calculates & stores the gradients for each model parameter in the parameters '.grad' attribute
loss.backward()

# An optimiser is then loaded, which is used to adjust the models internal parameters.
# In this case SDG is being used with a learning rate of 0.01 and momentum of 0.9, 
# with all model parameters being registered in the optimiser
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# '.step()' is then called to initiate gradient descent, 
# where the optimiser adjusts each parameter by the gradinent stored in its '.grad' attribute
optim.step()


# --- Autograd Details ---
# Above is all you need to know to start training nural networks, 
# the content below details how the undlying autograd function works, you may not need to know it for your usecase


# --- How Autograd Collects Gradients ---
# Say we create two tensors 'a' and 'b' with their 'requires_grad=ture',
# signling to autograd that every operation on them should be tracked

# defines tensors a & b
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

# Say we create another tensor 'Q' from tensors 'a' & 'b' where 'Q = 3a^3 - b^2' 
Q = 3*a**3 - b**2

# Assume that tensors 'a' & 'b' are to be parameters of a nural netork, with 'Q' being the loss/error.

# In nural network training, we want the gradient of the error with respect to the parameters, 
# ie '(σQ)/(σa) = 9a^2' and '(σQ)/(σb) = -2b', where σ is the derivation function (as in 'd' in 'd/dx')

# When the '.backward()' funciton is called on 'Q', 
# autogrand calculates the gradients specified above and stores them in each respective tensors '.grad' attribute

# Because 'Q.backward()' is a vector it requires that a 'gradient' argument be explicitly passed as a parameter, 
# where 'gradient' is a tensor of the same shape as 'Q' (the error), 
# and represents the gradient of 'Q' with respect to iteslf '(dQ)/(dQ) = 1' (the derivative of x w/ respect to x is 1).
# Note, equiviantly Q can be aggregated into a scalar and '.backward()' called implicitly 'Q.sum().backward()'

# represents gradient of Q
external_grad = torch.tensor([1., 1.])

# calculates gradients for each tensor used to calculate Q, stored in each tensors respective '.grad' attribute
Q.backward(gradient=external_grad) #eqivilant to Q.backward().sum()

# prints true if the derivatives / gradients manully calculated above are the same as the ones calculated by autograd
print(9*a**2 == a.grad)
print(-2*b == b.grad)


# --- Autograd DAG ---
# The directed acyclic graph (DAG) is how autograd keeps track of all changes made to tensors with 'requires_grad' set to true,
# and allows autograd to easily calculate the gradient/derivative of each tensor along the graph by using the gradient of the previous.

# For tensors that dont require gradients, setting 'requires_grad' to false will exclude it from the gradient computation DAG.
# The output tensor of an operation will have 'requires_grad' set to true if any of the input tensors used to produce it have 'requires_grad' set to true

# defining input tensors
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

# defining operation output tensors
a = x + y
b = x + z

print(f"b.requires_grad: {b.requires_grad}")
print(f"a.requires_grad: {a.requires_grad}")

# --- Frozen Parameters ---
# In a nural netowrk, parameters that dont have their gradients computed are ofter refered to as "frozen parameters".
# Freezing part of your model can be useful if you know in advace that you would need the gradients for the parameters you have frozen,
# doing this improves performance as it reduces the number of autograd calculations that need to be done

# Typically in finetuning, most of the model in frozen, with only the classifier layers being modified to make predictions on new labels,
# such as in the example below.

from torch import optim, nn

# creates instance of model with corresponding weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# freezes all parameters in network
for param in model.parameters():
    param.requires_grad = False

# Say we wanted of finetune the model on a new dataset with 10 labels.
# In resnet the classifier is the last linear layer, which is 'model.fc'.
# To do this it can simply be replaced by a new linear layer that acts as the models classifier (new linear layers are unfrozen by default)
model.fc = nn.Linear(512, 10)

# Now all the parameters in the model are frozen exept for 'model.fc', 
# meaning only gradients for the weights and biasis of 'model.fc' are computed.
# Even though all parameters are registerd in the optimizer, 
# only the gradients for the weights & biasis of the calsifier parameters are are being calculated and updated in gradient descent
# (note, torch.no_grad() can also be used to achive a simmilar functionaly to that listed above)  
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)


