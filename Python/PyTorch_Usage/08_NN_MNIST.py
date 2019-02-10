import torch
import torch.nn.init
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

mnist_train = dsets.MNIST(root='data/4_mnist',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='data/4_mnist',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
print(mnist_train.train_data.size())
print(mnist_train.train_labels.size())

idx=0
plt.imshow(mnist_train.train_data[idx,:,:].numpy(), cmap='gray')
plt.title('%i' % mnist_train.train_labels[idx])
plt.show()

batch_size=100
data_loader = data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=1)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

batch_images, batch_labels = next(iter(data_loader))

print(batch_images.size())
print(batch_labels.size())

# 숫자 100개 표시
imshow(utils.make_grid(batch_images))
batch_labels.numpy()

linear1 = torch.nn.Linear(784, 512)
linear2 = torch.nn.Linear(512, 10)
relu = torch.nn.ReLU()
model = torch.nn.Sequential(linear1, relu, linear2)

print(model)
cost_func = torch.nn.CrossEntropyLoss()

lr = 0.001
epochs = 5

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range(epochs):
    avg_cost = 0
    total_batch = len(mnist_train) // batch_size

    for batch_images, batch_labels in data_loader:
        X = Variable(batch_images.view(-1, 28*28))
        Y = Variable(batch_labels)
        optimizer.zero_grad()
        Y_prediction = model(X)
        cost = cost_func(Y_prediction, Y)
        cost.backward()
        optimizer.step()
        
        avg_cost += cost / total_batch

    print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_cost.data[0]))

print("Learning Finished!")

correct = 0
total = 0
for images,labels in mnist_test:
    images = Variable(images.view(-1, 28*28))
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += 1
    correct += (predicted == labels).sum()
    
print(correct / total)

r = random.randint(0, len(mnist_test)-1)
X_single_data = Variable(mnist_test.test_data[r:r + 1].view(-1,28*28).float())
Y_single_data = Variable(mnist_test.test_labels[r:r + 1])
single_prediction = model(X_single_data)
plt.imshow(X_single_data.data.view(28,28).numpy(), cmap='gray')
plt.show()

print('Label : ', Y_single_data.data.view(1).numpy())
print('Prediction : ', torch.max(single_prediction.data, 1)[1].numpy())

for i in range(20):
    weight = model[0].weight[i, :].data.view(28,28)
    weight = (weight - torch.min(weight))/(torch.max(weight)-torch.min(weight))
    print(i)
    plt.imshow(weight.numpy(), cmap='gray')
    plt.show()
