import pandas as pd

import torch
from torch import nn

from models import *

import matplotlib.pyplot as plt


# ---------------------------------------------------------------- #
# Load preprocessed Data                                           #
# ---------------------------------------------------------------- #

data = pd.read_csv('data/03_Building_Model_01_Parabolic_Data.csv')

# Avoid copy data, just refer
x = torch.from_numpy(data['x'].values).unsqueeze(1).float()
y = torch.from_numpy(data['y'].values).unsqueeze(1).float()

plt.xlim(-1.2, 1.2);    plt.ylim(-0.2, 1.3)
plt.scatter(x, y)
plt.title('03_Building_Model_01_Parabolic_Data')
plt.show()
# plt.savefig('output/figures/03_Building_Model_01_Parabolic_Data.png')


# ---------------------------------------------------------------- #
# Load Model                                                       #
# ---------------------------------------------------------------- #

model = TwoLayerNet(in_features=1, hidden_features=20, out_features=1)

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
    plt.xlim(-1.2, 1.2);    plt.ylim(-0.2, 1.3)
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'b--')
    plt.title('loss={:.4}'.format(loss.data.item()))
    plt.show()
    # plt.savefig('02_Linear_Regression_Model_trained.png')

display_results(model, x, y)


# ---------------------------------------------------------------- #
# Save Model                                                       #
# ---------------------------------------------------------------- #

torch.save(obj=model, f='03_Building_Model_01_Parabolic_Model.pt')

# ---------------------------------------------------------------- #
# Load and Use Model                                               #
# ---------------------------------------------------------------- #

loaded_model = torch.load(f='03_Building_Model_01_Parabolic_Model.pt')

display_results(loaded_model, x, y)
