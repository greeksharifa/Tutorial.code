import pandas as pd

import torch
from torch import nn
from torch.autograd import Variable

import matplotlib.pyplot as plt

"""
x = torch.arange(1, 11, dtype=torch.float).unsqueeze(1)
y = x / 2 + 1 + torch.randn(10).unsqueeze(1) / 5
print(x, y)

data = torch.cat((x, y), dim=1)
data = pd.DataFrame(data.numpy())

data.to_csv('data/02_Linear_Regression_Model_Data.csv', header=['x', 'y'])
"""

# ---------------------------------------------------------------- #
# Load preprocessed Data                                           #
# ---------------------------------------------------------------- #

data = pd.read_csv('data/02_Linear_Regression_Model_Data.csv')

# Avoid copy data, just refer
x = torch.from_numpy(data['x'].values).unsqueeze(1).float()
y = torch.from_numpy(data['y'].values).unsqueeze(1).float()

plt.xlim(0, 11);    plt.ylim(0, 8)
plt.scatter(x, y)
plt.title('02_Linear_Regression_Model_Data')
# plt.show()
plt.savefig('results/02_Linear_Regression_Model_Data.png')


# ---------------------------------------------------------------- #
# Load Model                                                       #
# ---------------------------------------------------------------- #

model = nn.Linear(in_features=1, out_features=1, bias=True)
"""
print(model)
print(model.weight)
print(model.bias)
"""

# ---------------------------------------------------------------- #
# Set loss function and optimizer                                  #
# ---------------------------------------------------------------- #

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(model(x))


# ---------------------------------------------------------------- #
# Train Model                                                      #
# ---------------------------------------------------------------- #

for step in range(501):
    prediction = model(x)
    loss = criterion(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ---------------------------------------------------------------- #
# Display output                                                   #
# ---------------------------------------------------------------- #

def display_results(model, x, y):
    prediction = model(x)
    loss = criterion(prediction, y)
    
    plt.clf()
    plt.xlim(0, 11);    plt.ylim(0, 8)
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'b--')
    plt.title('loss={:.4}, w={:.4}, b={:.4}'.format(loss.data.item(), model.weight.data.item(), model.bias.data.item()))
    plt.show()
    # plt.savefig('02_Linear_Regression_Model_trained.png')

display_results(model, x, y)


# ---------------------------------------------------------------- #
# Save Model                                                       #
# ---------------------------------------------------------------- #

torch.save(obj=model, f='02_Linear_Regression_Model.pt')

# ---------------------------------------------------------------- #
# Load and Use Model                                               #
# ---------------------------------------------------------------- #

loaded_model = torch.load(f='02_Linear_Regression_Model.pt')

display_results(loaded_model, x, y)
