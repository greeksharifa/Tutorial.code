import os
import sys

import torch
from torch.autograd import Variable

sys.path.append(os.getcwd())
import time_check

CheckTime = time_check.CheckTime()

a = torch.rand(10000, 10000)
b = torch.rand(10000, 10000)

CheckTime.start()
a.matmul(b)
CheckTime.end()

A = a.cuda()
B = b.cuda()

CheckTime.start()
A.matmul(B)
CheckTime.end()

######################################################################

x = Variable(torch.ones(2, 2), requires_grad=True)
y = x.mean()

y.backward()

x.grad
x.grad_fn
x.data
y.grad_fn

# Training Data

def get_data():
    train_X = np.ararray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                          7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                          2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    dtype = torch.FloatTensor
    X = Variable(torch.from_numpy(train_X).type(dtype), requires_grad=False).view(17, 1)
    y = Variable(torch.from_numpy(train_y).type(dtype), requires_grad=False)

    return X, y

def get_weights():
    w = Variable(torch.randn(1), requires_grad=True)
    b = Variable(torch.randn(1), requires_grad=True)
    return w, b

def simple_network(x):
    y_pred = torch.matmul(x, w) + b
    return y_pred

f = torch.nn.Linear(17, 1)

def loss_fn(y, y_pred):
    loss = (y - y_pred).pow(2).sum()
    for param in [w, b]:
        if not param.grad is None: param.grad.data.zero_()
    loss.backward()

    return loss.data[0]

def optimize(learning_rate):
    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data

from torch.utiis.data import Dataset, DataLoader

import glob

class DogsAndCatsDataset(Dataset):
    def __init__(self, root_dir, size=(224, 224)):
        self.files = glob.glob(root_dir)
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img = np.asarray(Image.open(self.files[idx]).resize(self.size))
        label = self.files[idx].split('/')[-2]
        return img, label

dataloader = DataLoader(dogsdset, batch_size=32, num_workers=2)
for imgs, labels in dataloader:
    # Apply DL on the dataset.
    pass

def plot_variable(x, y, z='', **kwargs):
    l = []
    for a in [x, y]:
        if type(a) == Variable:
            l.append(a.data.numpy())
    plt.plot(l[0], l[1], z, **kwargs)
