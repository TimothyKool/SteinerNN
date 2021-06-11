import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import csv
from tqdm import tqdm

# Load data
x = np.zeros((512, 10000), dtype=np.float32)
y = np.zeros((512, 10000), dtype=np.float32)

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
        self.layers = nn.Sequential(
            nn.Linear(10000, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10000),
            nn.Hardtanh()
        )

    def forward(self, x):
        return self.layers(x)

# 60% 20% 20%
model = Net()
my_loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
for j in tqdm(range(20)):
    for i in range(512):
        x_tensor = torch.from_numpy(x[i])
        y_tensor = torch.from_numpy(y[i])
        # calls forward function on x_tensor
        output = model(x_tensor)

        loss = my_loss(output, y_tensor)
        
        if i % 50 == 0:
            print(i, loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Load in new data set
# Evaluate model
for i in range(512):
    x_tensor = torch.from_numpy(x[i])
    y_tensor = torch.from_numpy(y[i])
    # calls forward function on x_tensor
    output = model(x_tensor)

    loss = my_loss(output, y_tensor)
    
    if i % 50 == 0:
        print(i, loss)