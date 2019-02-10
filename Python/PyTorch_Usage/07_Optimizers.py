import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))
plt.scatter(x.numpy(), y.numpy())
plt.show()

torch_dataset = data.TensorDataset(x, y)
loader = data.DataLoader(dataset=torch_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         num_workers=2,)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(1, 20)
        self.predict = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net_SGD = Net()
net_Momentum = Net()
net_RMSProp = Net()
net_Adam = Net()
nets = [net_SGD, net_Momentum, net_RMSProp, net_Adam]

opt_SGD  = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
#같은 방향으로 계속 움직이는 경향이 있습니다
opt_RMSProp = torch.optim.RMSprop(net_RMSProp.parameters(),lr=LR, alpha=0.9)
#과거 그라디언트의 지수 함수 적으로 감쇠 평균 유지
opt_Adam  = torch.optim.Adam(net_Adam.parameters(),lr=LR, betas=(0.9,0.99))
optimizers = [opt_SGD, opt_Momentum,opt_RMSProp,opt_Adam]

criterion = nn.MSELoss()
loss_history = [[],[],[],[]]
for epoch in range(EPOCH):
    print('epoch:', epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        for net, opt, l_his in zip(nets, optimizers, loss_history):
            output = net(b_x)
            loss = criterion(output, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data[0])

labels = ['SGD', 'Momentum','RMSprop','Adam']
for i, l_his in enumerate(loss_history):
    plt.plot(l_his, label = labels[i])
plt.legend(loc='best')
plt.xlabel('steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()
