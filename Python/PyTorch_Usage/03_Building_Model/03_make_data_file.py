import os

import pandas as pd

import torch


# ---------------------------------------------------------------- #
# Make Parabolic Data                                              #
# ---------------------------------------------------------------- #

x = torch.linspace(-1, 1, 101).unsqueeze(dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
# print(x, y)

data = torch.cat((x, y), dim=1)
data = pd.DataFrame(data.numpy())

os.makedirs('data', exist_ok=True)
data.to_csv('data/03_Building_Model_01_Parabolic_Data.csv', header=['x', 'y'])


# ---------------------------------------------------------------- #
# Make Parabolic Data                                              #
# ---------------------------------------------------------------- #
