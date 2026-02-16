import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
import pandas as pd
from torchvision.io import decode_image

from torch.utils.data import DataLoader

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}

figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1, )).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(img.squeeze(), cmap="grey")
plt.show()


#in this example, images are stored in image dir, and their labels are stored sparately in a CSV annotations_file
class CustomImageDataset(Dataset):
    
    # runs once when initialising the dataset object, must initialise the directory containing the images, the anotations file, and both transforms
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    # returns the number of samples in a dataset
    def __len__(self):
        return len(self.img_labels)
    
    # Loads and returns the sample in the dataset at the given index, it works by using the index to retrive the image, 
    # converting it to a tensor using decode_image, retriving the corresponding label from the csv data in self.img_labels,
    # calls the transform function on them (if not set to none), and returns the tensor image with its corresponding label in a tuple
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# the contents of the csv file should look like
# tshirt1.jpg, 0
# tshirt2.jpg, 0
# ......
# ankleboot999.jpg, 9

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

#display image and label
train_features, train_labels = next(iter(train_dataloader))
print(f'F Batch Shape: {train_features.size()}')
print(f'L Batch Shape: {train_labels.size()}')
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
print(f"Label: {label}")
plt.show()
