import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(4)
import numpy as np
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)
x = Variable(x)
x_np = x.data.numpy()
y_relu = F.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()
plt.figure(1, figsize=(8,6))

plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1,5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')
plt.show()

x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
x,y = Variable(x), Variable(y)
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

# model

hidden = nn.Linear(1, 20, bias=True)
activation = nn.ReLU()
output = nn.Linear(20, 1, bias=True)

net = nn.Sequential(hidden, activation, output)
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
criterion = torch.nn.MSELoss()

x.size()

for t in range(200):
    prediction = net(x)

    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 10 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r--', lw=6)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.show()
        plt.pause(0.2)

plt.ioff()

torch.save(net.state_dict(), 'nonlinear.pkl')
net.state_dict()

net.load_state_dict(torch.load('nonlinear.pkl'))
net.state_dict()

torch.save(net,'net.pkl')
net = torch.load('net.pkl')
