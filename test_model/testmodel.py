import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

mainDataset = datasets.Caltech101(
    root="./data_test_model",
    download=True,
    transform=transform
)

train_dataset_size = int(0.8 * len(mainDataset))
test_dataset_size = len(mainDataset) - train_dataset_size

train_dataset, test_dataset = random_split(mainDataset, [train_dataset_size, test_dataset_size])

dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=32, shuffle=True)

class neuralNetwork(nn.Module):
    def __init__(self, numClasses = 101):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, numClasses)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # 224 → 112
            x = self.pool(F.relu(self.conv2(x)))  # 112 → 56
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x


model = neuralNetwork().to("cpu")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader_train:
        images, labels = images.to("cpu"), labels.to("cpu")

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader_train):.4f}")