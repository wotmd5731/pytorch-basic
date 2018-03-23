# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import matplotlib.pylab as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, D_in,h, D_out):
        super().__init__()
        self.l1 = torch.nn.Linear(D_in, h)
        self.l2 = torch.nn.Linear(h, h)
        self.l3 = torch.nn.Linear(h, h)
        self.l4 = torch.nn.Linear(h, D_out)

    def forward(self, x):
        x = F.relu6(self.l1(x))
        x = F.relu6(self.l2(x))
        x = F.relu6(self.l3(x))
        x = F.relu6(self.l4(x))
        return x


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1, 1, 100, 1

# Create random Tensors to hold inputs and outputs, and wrap them in Variables
x = Variable(torch.randn(N, D_in), requires_grad=False)
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Construct our model by instantiating the class defined above
model = Net(D_in,H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=10)



for t in range(20):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)
#    model.l1.weight
    nn.init.normal(model.l1.weight)
    nn.init.normal(model.l1.bias)
    nn.init.normal(model.l2.weight)
    nn.init.normal(model.l2.bias)
    nn.init.normal(model.l3.weight)
    nn.init.normal(model.l3.bias)
    nn.init.normal(model.l4.weight)
    nn.init.normal(model.l4.bias)
#    w = list(model.parameters())
#    print(w)
    # Compute and print loss
#    loss = criterion(y_pred, y)
#    print(t, loss.data[0])


    tx = Variable(torch.linspace(-10,10), requires_grad=False).unsqueeze(1)
    ty = model(tx)
    plt.plot(tx.view(-1).data.numpy(),ty.view(-1).data.numpy())
    plt.pause(0.1)


    # Zero gradients, perform a backward pass, and update the weights.
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()


