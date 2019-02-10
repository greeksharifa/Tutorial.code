import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(5)

n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1)
#평균과 표준 편차가 주어진 별도의 정규 분포에서 추출한 난수의 텐서 (Tensor)를 반환합니다.
y0 = torch.zeros(100,1)
x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100,1)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), 0).type(torch.FloatTensor)
x,y =  Variable(x), Variable(y)

torch.cat( (x, y), 1)

# logistic model

linear = nn.Linear(2, 1, bias=True)
model = nn.Sequential(linear, nn.Sigmoid())
model.state_dict()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.BCELoss() # cost_fn = -(y * torch.log(prob) + (1 - y)* torch.log(1 - prob) ).mean()

x.size()

for t in range(120):
    prob = model(x)
    cost = criterion(prob, y)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

plt.cla()
prediction = prob.gt(0.5)
pred_y = prediction.data.numpy().squeeze()
target_y = y.data.squeeze(1).numpy()

x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min.data.numpy(), x_max.data.numpy(), 1),
                     np.arange(y_min.data.numpy(), y_max.data.numpy(), 1))

Z = model(Variable(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()])).float())

Z = Z.view(xx.shape)
plt.contourf(xx, yy, Z.data.numpy(), cmap=plt.cm.binary)

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='seismic')
# RdYlGn
accuracy = sum(pred_y == target_y) / 200.
plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'Blue'})
plt.show()
plt.pause(0.1)



## Softmax Regression

torch.manual_seed(5)
nb_classes = 3


x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5],
          [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
          [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

X = Variable(torch.Tensor(x_data))
Y = Variable(torch.Tensor(y_data))

_, Y_label = Y.max(dim=1)
# 1 - - - 부분중 어디가 제일 큰지 즉 1인 부분을 찾는 방법
print(Y_label)

linear = nn.Linear(4, 3)
model  = nn.Sequential(linear)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()
for step in range(100001):
    prediction = model(X)
    cost = criterion(prediction, Y_label)
    optimizer.zero_grad()

    cost.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(step, cost.data.numpy())

# Testing & One-hot encoding
print('--------------')
a = model(Variable(torch.Tensor([[1, 11, 7, 9]])))
print(a.data.numpy(), torch.max(a, 1)[1].data.numpy())

print('--------------')
b = model(Variable(torch.Tensor([[1, 3, 4, 3]])))
print(b.data.numpy(), torch.max(b, 1)[1].data.numpy())

print('--------------')
c = model(Variable(torch.Tensor([[1, 1, 0, 1]])))
print(c.data.numpy(), torch.max(c, 1)[1].data.numpy())

print('--------------')
all = model(Variable(torch.Tensor([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]])))
print(all.data.numpy(), torch.max(all, 1)[1].data.numpy())
