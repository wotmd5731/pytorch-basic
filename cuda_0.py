# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import time
#13.454618692398071

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 256, 1024, 1024, 10
cuda = True
if cuda:
    xx = torch.randn(N, D_in).type(torch.cuda.FloatTensor)
    yy = torch.randn(N, D_out).type(torch.cuda.FloatTensor)
else:
    xx = torch.randn(N, D_in)
    yy = torch.randn(N, D_out)
    
# Create random Tensors to hold inputs and outputs, and wrap them in Variables
x = Variable(xx, requires_grad=False)
y = Variable(yy, requires_grad=False)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)
if cuda:
    model.cuda()
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

s_t = time.time()
for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

e_t = time.time()
print(e_t-s_t)
