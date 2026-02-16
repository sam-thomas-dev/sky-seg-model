import torch
import torch.nn as nn
import torch.nn.functional as F

# --- NN Definition ---
# Outlined below is the definition for a convolutional nural nework (the backward function is defined implicitly using autograd)

# When defining the forward and backward functions, 
# you only need to define the forward function,
# as the backward function is automatically defining using autograd.
# You can use any tensor functions in the forward function.
 
# Kernel: a small matrix of weights that slides over the input to detect specific local patterns/features
# Affine Opperation: a linear transformation followed by a shift, where the amount shifted corresponds to the bias
# Convolution Layer: automatically extracts useful features from structured data, mainly to detect local patterns
# Subsampling / Pooling Layer: downsamples feature maps to make the network more efficient and more robust to small spacial changes
# Activation Function: decides how strongly a neuron fires by transforming the weighted sum of its inputs
# RELU: a rectified linear unit is an activation function, used in nural networks to introduce non-linearity
# Flatten operation: reshapes multidimentional data into a onedimentional vector without changing the data values themselves
# Fully Connected Layer: performs an affine opperation on a flattened input to produce high-level predictions / decisions

# torch.nn.Conv2d(x, y, z): defines a 2d convolution layer typically used for images, where x = in_channels, y = out_channels, and z = kernel_size
#   so in the case of nn.Conv2d(1, 6, 5), you are defining a 2d convolution layer with 1 in channel (which in the case of an image would mean that its grey scale), 
#   6 out channels (meaning the layer will learn 6 different filters, with each filter producing 1 filter map, producing an output with 6 channels, 
#   think of it as instructing the nn to "look at the image in 6 different learned ways"), and a kernel size of 5x5 (meaning each filter is 5x5, 
#   so the convolution layer will slide a 5x5 window over the image, meaning each filter learns 25 weights + optional bias).

# torch.nn.Linear(x, y): defines a fully conennected (dense) layer that perfroms a linear (affine) transformation, where x = in_features, and y = out_features or neurons
#   so in the case of nn.Linear(400, 120), it will take all 400 input values, multiply them by learned weights, sum them and add bias, 
#   with this being done 120 times, once per output neuron

# torch.nn.MaxPool2d(x, y): defines a 2d max pooling layer, which downsamples feature maps by keeping only the largest (important) values within a small region, 
#   where x = kernel_size, and y = stride, do in the case of nn.MaxPool2d(2, 2) would take a 2x2 window, find the maximum value in that window, output that value,
#   and move the window down by 2 pixels (because stride = 2)

# torch.flatten(x, y): reshapes the output of a convolution layer so it can be fed into a fully connected layer without mixing up batch dimentions, 
#   where x = the output of the convolution layer, and y = start_dim, so in the case of torch.flatten(x, 1), 
#   the function will output x with every dimention starting from dimention index 1, flattened into one dimention, 
#   meaning dimention 0 is excluded (the batch dimentions) 

# torch.nn.functional.relu(x): applies the ReLU activation function to an output to introduce non-linearity into the network, 
#   doing this by setting values to 0 if they are negative, and keeping positive values the same, 
#   where x = the output from a convolution layer or fully connected layer. Because the outputs conv & fc layers are linear, 
#   if their output were to be fed into another fc or conv layer, the model might pick up on the linear relationship insead of the actual useful patterns, 
#   introducing non-linearity into the output with an activation function like ReLU (others user different methods to achive this) prevents this from occuring.
#   The final output produced by your forward function should not have been fed through an activation function, it should be linear


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input image channel, 6 output channels, 5x5 square convolution
        
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5) #defines convolution layer 1
        self.conv2 = nn.Conv2d(6, 16, 5) #defines convolution layer 2

        # an affine opperation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120) #(5x5 from image dimention), defines fully connected layer 1
        self.fc2 = nn.Linear(120, 84) #defines fully connected layer 2
        self.fc3 = nn.Linear(84, 10) #defines fully connected layer 3

    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function,
        # and outputs tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1 = F.relu(self.conv1(input))

        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 6, 14, 14) tensor
        s2 = F.max_pool2d(c1, (2, 2))

        # Convolution layer C3: 6 input channels, 16 output channels, 5x5 square convolution,
        # it uses RELU activation function, and outputs a (N, 16, 10, 10) tensor
        c3 = F.relu(self.conv2(s2))

        # Subsampling layer S4: 2x2 grid, purely functional, this layer does not have any parameter,
        # and outputs a (N, 16, 5, 5) tensor 
        s4 = F.max_pool2d(c3, 2)
        
        # Flatten operation: purely functional, outputs (N, 400) tensor
        s4 = torch.flatten(s4, 1)

        # Fully connected layer F5: (N, 400) tensor input, and outputs a (N, 120) tensor, 
        # it uses RELU activation function
        f5 = F.relu(self.fc1(s4))

        # Fully connected layer F6: (N, 120) tensor input, and outputs (N, 84) tensor,
        # it uses RELU activation function
        f6 = F.relu(self.fc2(f5))

        # Fully connect layer OUTPUT: (N, 84) tensor input, and outputs a (N, 10) tensor
        output = self.fc3(f6)
        return output

# creats instance of NN defined above    
net = Net()

# print(net)


# --- NN Parameters ---
# the learnable parameters of the model are returned by the '.parameters()' attribute

# get parameters from net as list
params = list(net.parameters())

#conv1s .weight attribute
conv1s_weight = params[0].size()

# print(len(params))
# print(conv1s_weight)

# Note the expected input size of this net is 32x32, images will need to be resized to meed this exprectation before use
# Below is an example of a random 32x32 input

# creats random input tensor
input = torch.randn(1, 1, 32, 32) 

# captures NN output given input defined above
out = net(input)

# print(out)

# zeros gradient buffers of all parameters 
net.zero_grad()

# backward propogates with random gradients
out.backward(torch.randn(1, 10))

# Note, the entire torch.nn package only supports inputs that are a mini-batch of samples, not a single sample.
# For example, in the nn defined above, nn.Conv2d will take a tensor with 4 dimentions 'nSamples x nChannels x Height x Width'.
# If you have a single sample, you can use the 'input.unsqueeze(0)' method to add a fake batch dimention


# --- Class Definitions ---
# torch.Tensor: a multidimentional array with support for autograd operations like backward().
# also holds the gradient with with respect to the tensor.

# nn.Module: neural network module, a convinient way of encapsulating parameters, 
# with helpers for exporting, moving to GPU, loading, etc.

# nn.Parameter: a kind of tensor that is automatically registered as a parameter when assigned as an attribute to a module

# autograd.Function: implements forward and backward definitions of an autograd operation. 
# Every tensor operation creates at least a single Function node that connects to functions that created a tensor and encodes its history


# --- Loss Function ---
# A loss function takes the input pair output & target, and computes a value that estimates how far away the output is from the target
# The nn package povides a variety of different loss functions, 
# one of which is 'nn.MSELoss()' which computes the mean squared error between the output and target inputs

# calculates network output given spcified input
output = net(input)

# defines dummy target
target = torch.randn(10)

# makes target same shape as output
target = target.view(1, -1)

# creates new MSELoss function
criterion = nn.MSELoss()

# calculates MSE between output and target 
loss = criterion(output, target)

# print(loss)

# if you follow 'loss' in the backward direction using the '.grad_fn' attribute to go backward up the graph, 
# you will see a graph of computations that looks like the following: 
#   input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#         -> flatten -> linear -> relu -> linear -> relu -> linear
#         -> MSELoss
#         -> loss

# When you call loss.backward(), the whole graph is diferentiated with respect to the nural network parameters, 
# so all tensors in the graph with requires_grad set to true will have the value in their .grad attribute accumulated into the gradient.
# To show this, here is an example manually going a few steps backward.

#MSELoss Function
# print(loss.grad_fn)

#Linear Function
# print(loss.grad_fn.next_functions[0][0])

#ReLu Function
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) 


# --- Backward Propogation ---
# To backward propogate all we have to do is call the '.backward()' method.
# However, fist you need to clear the existing gradients, otherwise the gradients will be acumulated with the existing gradients
# In the following example, '.backward()' is called on 'loss', 
# with conv1's bias gradient being printed both before and after the function call

# zeros the gradient buffers of all parameters
net.zero_grad()

print(f"conv1.bias.gard before: {net.conv1.bias.grad}")

loss.backward()

print(f"conv1.bias.gard after: {net.conv1.bias.grad}")

# for a full list of avalible modules and loss functions, refer to the pytorch torch.nn documentation 


# --- Updating Weights ---
# The simplest rule for updating weights used in practice is the socastic gradient descent (SGD),
# deffined as following: weight = weight - learning_rate * gradient, 
# the following is a simple implementation of said definition

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# However, you many want to implement diferent more complex weight update methods such as SGD, Nesterov-SDG, Adam, or RMSProp.
# The simplest way to do this is using the built-in 'torch.optim' package which implements these methods.
# Below is an example using this package.

import torch.optim as optim

# creates optimizer with predefined SGD upate rule method
optimizer = optim.SGD(net.parameters(), lr=0.01)

# following code should be in training loop for model
optimizer.zero_grad() # zeros gradient buffers for all parameters exposed to optimiser
output = net(input) # generates model output from specified input (forward propgation)
loss = criterion(output, target) # calculates MSE for diference between output and target
loss.backward() # calculates gradients for all model parameters (backward propogation)
optimizer.step() # performs weight updates based on gradients generated during backward propogation 

# Note, gradient buffers MUST manually be set to zero using optimizer.zero_gard() as gradients are acumulated

