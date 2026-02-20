import torch
import torch.nn as nn

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

        # returned logits
        return self.outputConv(d1)
    
