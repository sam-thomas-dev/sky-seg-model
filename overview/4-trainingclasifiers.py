import torch
import torchvision
import torchvision.transforms as transforms
#---
import matplotlib.pyplot as plt
import numpy as np
#---
import torch.nn as nn
import torch.nn.functional as F
#---
import torch.optim as optim


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#---

# functions to show image
def imshow(img):
    img = img / 2 + 0.5 #unnormalizes
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#---

# # gets some random training data
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# # show images
# imshow(torchvision.utils.make_grid(images))

# #prints image labels
# print(' '.join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))

#---

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #Takes 3 input channels from image, turns it into 6 output channels, with a kernel size of 5x5
        self.conv1 = nn.Conv2d(3, 6, 5) 
        #Takes 6 output channels from previous convolution layer as input channels, turns it into 16 output channels, with a kernel size of 5x5
        self.conv2 = nn.Conv2d(6, 16, 5) 

        #Downsamples feature map keeping the largest value in each 2x2 area of the image / map
        self.pool = nn.MaxPool2d(2, 2) 
        
        #Takes output features of final convolution layer as input features (16 output channels over a 5x5 dimention, so 400 input features), 
        #transforms them into 120 output features/neurons
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        #Takes 120 output features of previous fully connected layer as input features, and transforms them into 84 output features/neurons
        self.fc2 = nn.Linear(120, 84)
        #Takes 84 output features of previous fully connected layer as input features, and transforms them into 10 output features/neurons (which maps onto our 10 classes)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        #Calls convolution layer 1 on x input, 
        #output of conv1 is fed through ReLU activation function making it non-linear allowing it to be used as the input for another layer, 
        #output of activation function is fed through downsampler where the largest value is kept from each 2x2 section of the output
        x = self.pool(F.relu(self.conv1(x)))
        #Calls convolution layer 2 on ouput of convolution layer 1,
        #output of conv1 is fed through ReLU activation function making it non-linear allowing it to be used as the input for another layer, 
        #output of activation function is fed through downsampler where the largest value is kept from each 2x2 section of the output
        x = self.pool(F.relu(self.conv2(x)))
        #Reshapes output of convolution layer 2 so it can be used as input for fully connected layer 1
        x = torch.flatten(x, 1) # flattens all dimentions exept batch
        #Calls fully connected layer 1 on reshaped output of convolution layer 2,
        #output of fully connected layer 1 is fed through ReLU activation function making it non-linear allowing it to be used as the input for another layer
        x = F.relu(self.fc1(x))
        #Calls fully connected layer 2 on output of fully connected layer 1,
        #output of fully connected layer 2 is fed through ReLU activation function making it non-linear allowing it to be used as the input for another layer
        x = F.relu(self.fc2(x))
        #Calls fully connected layer 3 on output of fully connected layer 2
        x = self.fc3(x)
        #Returns output of fully connected layer 3
        return x
    
#---

# net = Net()

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# for epoch in range(2): #loop over data 2 times
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         #get the inputs, data is a list of [inputs, labels]
#         inputs, labels = data
        
#         #zero the parameter gradients
#         optimizer.zero_grad()

#         #forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         #print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999: #print every 2000 mini batches
#             print(f'[{epoch + 1}, {i+1:5d}] loss: {running_loss / 2000:.3f}')

# print('Finished Training.')

PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)

#---

dataiter = iter(testloader)
images, labels = next(dataiter)

#print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

#---

net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))


# correct = 0
# total = 0
# # since were not training the gradients dont need to be calculated for outputs
# with torch.no_grad():
#     for data in testloader:
#         images, labes = data
#         # calculate outputs by running images through network
#         outputs = net(images)
#         # class with highest energy is chosen as prediction
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Accuracy of network on 10000 test images: {100 * correct // total}%')

#prepare to count preditions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

#again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        #collect the prediections for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

#print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is  {accuracy:.1f} %')

if torch.cuda.is_available():
    device = torch.device('cuda:0')

    net.to(device)
    #remember that you will need to send inputs and targets at every step to the gpu
    inputs, labels = data[0].to(device), data[1].to(device)