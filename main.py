import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import csv

# Load data
x = np.zeros((512, 10000))
y = np.zeros((512, 10000))

with open("./smt100_ds11776/ps100_0_511.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    it = int(0)
    for line in reader:
        # Empty row means that the set of points for the current graph
        # has ended. Update the iterator and continue
        if(len(line) == 0):
            it = it + 1
            continue
        # Skip over the starting row of each graph that only has the graph
        # number and number of steiner points
        if(len(line) == 2):
            continue
        # Coordinates of the current row
        row = int(line[1])
        column = int(line[2])
        # Fill in X and Y
        if(int(line[3]) == 0):
            x[it][100*row + column] = -1.0
        y[it][100*row + column] = 1.0

# Create model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_body = 3

        self.act = torch.nn.ReLU(inplace=True)
        self.head = nn.Conv2d(1, 8, 3, stride=1, padding=1, bias=True)
        self.body = nn.Sequential([(nn.Conv2d(8, 8, 3, stride=1, padding=1, bias=True), self.act) for it in range(num_body)])
        self.tail = nn.Conv2d(8, 1, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        y = self.act(self.head(x))
        y = self.body(y)
        y = self.act(self.tail(y))

        return y

def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss

model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
for i in range(400):
    x_tensor = torch.from_numpy(x[i])
    y_tensor = torch.from_numpy(y[i])

    output = model(x_tensor)

    loss = my_loss(output, y_tensor)
    if i % 50 == 0:
        print(i, loss)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')


