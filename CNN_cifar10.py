import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#0. device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1. hyperparameters
num_epoch = 15
batch_size = 4
learning_rate = 0.001

#2. dataset
##dataset has PILImage images of range [0,0]
##we tranform them to Tensors of normalized range[-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_dataset = torchvision.datasets.CIFAR10(root="./CIFAR10",train=True,download=True,transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="./CIFAR10",train=False,download=True,transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

Classes =("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")

#3.Conv Net implementation
class Convnet(nn.Module):
    def __init__(self):
        super(Convnet, self).__init__()
        #the inpute channel(colors), output channel, kernel
        self.Conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.Conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x = self.pool(F.relu(self.Conv1(x)))
        x = self.pool(F.relu(self.Conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Convnet().to(device)

#4. loss and optimization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#6. Training the model
n_total_steps = len(train_loader)
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        #orgin shape of :[4,3,32,32] =4,3,1024
        #input_layer :3 input channels and 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward pass : gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i +1) % 2000 ==0:
            print(f"epoch:{epoch+1}/{num_epoch}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}")
print("training finished")

#7. testing and evaluation
# in the testing phase, we don't need to compute gradients for memory efficiency
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        #value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predictions[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    acc = 100.0 * n_correct/ n_samples
    print(f"accuracy={acc}")

    for i in range(10):
        acc = 100.0 * n_class_correct[i]/n_class_samples[i]
        print(f"Accuracy of {Classes[i]}:{acc}%")


