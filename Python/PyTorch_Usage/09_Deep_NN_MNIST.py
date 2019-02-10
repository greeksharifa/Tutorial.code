import torch
import torch.nn.init
from torch.autograd import Variable

import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import random

mnist_train = dsets.MNIST(root='data/4_mnist',
                         train=True,
                         transform = transforms.ToTensor(),
                         download=True)

mnist_test = dsets.MNIST(root='data/4_mnist',
                        train=False,
                        transform=transforms.ToTensor(),
                        download=True)

batch_size = 100

data_loader =  torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=1)

linear1 = torch.nn.Linear(784, 512, bias=True)
linear2 = torch.nn.Linear(512, 512, bias=True)
linear3 = torch.nn.Linear(512, 512, bias=True)
linear4 = torch.nn.Linear(512, 512, bias=True)
linear5 = torch.nn.Linear(512, 256, bias=True)
linear6 = torch.nn.Linear(256, 128, bias=True)
linear7 = torch.nn.Linear(128, 10, bias=True)
relu = torch.nn.ReLU()

model = torch.nn.Sequential(linear1, relu,
                           linear2, relu,
                           linear3, relu,
                           linear4, relu,
                           linear5, relu,
                           linear6, relu,
                           linear7, relu)

print(model)

cost_func = torch.nn.CrossEntropyLoss()

lr = 0.001
training_epochs = 30
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(mnist_train) // batch_size
    
    for images, labels in data_loader:
        X = Variable(images.view(-1, 28*28))
        Y = Variable(labels)
        
        optimizer.zero_grad()
        Y_prediction = model(X)
        cost = cost_func(Y_prediction, Y)
        cost.backward()
        optimizer.step()
        
        avg_cost += cost / total_batch
    print("[Epoch: {:>4}] cost  = {:>.9}".format(epoch + 1, avg_cost.data[0]))

print('Finished')

correct = 0
total = 0
for images, labels in mnist_test:
    images = Variable(images.view(-1, 28 * 28))
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += 1
    correct += (predicted == labels).sum()

print('Accuracy test on the 10000 images : %d %%' % (100 * correct / total))

r = random.randint(0, len(mnist_test) -1)
X_single_data = Variable(mnist_test.test_data[r:r + 1].view(-1,28 * 28).float())
Y_single_data = Variable(mnist_test.test_labels[r:r + 1])

single_prediction = model(X_single_data)

print("Label : ", Y_single_data.data)
print("Prediction : ", torch.max(single_prediction.data, 1)[1])

plt.imshow(X_single_data.data.view(28,28).numpy(), cmap='gray')
plt.show()

torch.save(model.state_dict(), 'DNN.pkl')