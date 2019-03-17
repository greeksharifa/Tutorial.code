import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import argparse

from matplotlib import pyplot as plt
import numpy as np

import pickle
import os
import imageio


parser = argparse.ArgumentParser(description='GAN tutorial: MNIST')

parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--batch-size', type=int, default=64, help='size of mini-batch')
parser.add_argument('--noise-size', type=int, default=100, help='size of random noise vector')
parser.add_argument('--use-cuda', type=bool, default=True, help='use cuda if available')
parser.add_argument('--learning-rate', '-lr', type=float, default=0.0002, help='learning rate of AdamOptimizer')
parser.add_argument('--beta1', type=float, default=0.5, help='parameter beta1 of AdamOptimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='parameter beta2 of AdamOptimizer')
parser.add_argument('--output-dir', type=str, default='output/', help='directory path of output')
parser.add_argument('--log-file', type=str, default='log.txt', help='filename of logging')

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

use_cuda = args.use_cuda and torch.cuda.is_available()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

mnist = datasets.MNIST(root='data', download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=args.batch_size, shuffle=True)


class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(in_features=100, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=512)
        self.linear3 = nn.Linear(in_features=512, out_features=1024)
        self.linear4 = nn.Linear(in_features=1024, out_features=28 ** 2)
    
    
    def forward(self, x):
        """
        :param x: input tensor[batch_size * noise_size]
        :return: output tensor[batch_size * 1 * 28 * 28]
        """
        x = nn.LeakyReLU(0.2)(self.linear1(x))
        x = nn.LeakyReLU(0.2)(self.linear2(x))
        x = nn.LeakyReLU(0.2)(self.linear3(x))
        x = nn.Tanh()(self.linear4(x))
        return x.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(in_features=28 ** 2, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=512)
        self.linear3 = nn.Linear(in_features=512, out_features=256)
        self.linear4 = nn.Linear(in_features=256, out_features=1)
    
    
    def forward(self, x):
        """
        :param x: input tensor[batch_size * 1 * 28 * 28]
        :return: possibility of that the image is real data
        """
        x = x.view(-1, 28 ** 2)
        x = nn.LeakyReLU(0.2)(self.linear1(x))
        x = nn.Dropout()(x)
        x = nn.LeakyReLU(0.2)(self.linear2(x))
        x = nn.Dropout()(x)
        x = nn.LeakyReLU(0.2)(self.linear3(x))
        x = nn.Dropout()(x)
        return nn.Sigmoid()(self.linear4(x))


G = Generator()
D = Discriminator()

if use_cuda:
    G.cuda()
    D.cuda()

criterion = nn.BCELoss()
G_optimizer = Adam(G.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
D_optimizer = Adam(D.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))


def square_plot(data, path):
    """Take an array of shape (n, height, width) or (n, height, width , 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    if type(data) == list:
        data = np.concatenate(data)
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imsave(path, data, cmap='gray')


train_hist = {'D_losses': [], 'G_losses': []}
generated_images = []

z_fixed = torch.randn(25, args.noise_size)
if use_cuda: z_fixed = z_fixed.cuda()

with open(args.output_dir + args.log_file, 'w', encoding='utf-8') as log_f:
    for epoch in range(args.epochs):
        
        D_losses = []
        G_losses = []
        
        for D_real_data, _ in dataloader:
            
            batch_size = D_real_data.size(0)
            
            # Training D with real data
            D.zero_grad()
            
            target_real = torch.ones(batch_size, 1)
            target_fake = torch.zeros(batch_size, 1)
            
            if use_cuda:
                D_real_data, target_real, target_fake = \
                    D_real_data.cuda(), target_real.cuda(), target_fake.cuda()
            
            D_real_decision = D(D_real_data)
            D_real_loss = criterion(D_real_decision, target_real)
            
            # Training D with fake data
            
            z = torch.randn((batch_size, args.noise_size))
            if use_cuda: z = z.cuda()
            
            D_fake_data = G(z)
            D_fake_decision = D(D_fake_data)
            D_fake_loss = criterion(D_fake_decision, target_fake)
            
            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            
            D_optimizer.step()
            
            # Training G based on D's decision
            G.zero_grad()
            
            z = torch.randn((batch_size, args.noise_size))
            if use_cuda: z = z.cuda()
            
            D_fake_data = G(z)
            D_fake_decision = D(D_fake_data)
            G_loss = criterion(D_fake_decision, target_real)
            G_loss.backward()
            
            G_optimizer.step()
            
            # logging
            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())
        
        tpr = (D_real_decision > 0.5).float().mean().item()
        tnr = (D_fake_decision < 0.5).float().mean().item()
        log = "Epoch: {epoch:<3d} \t D Loss: {d_loss:<8.4} \t G Loss: {g_loss:<8.4} \t " \
              "True Positive Rate: {tpr:<6.1%} \t True Negative Rate: {tnr:<6.1%}".format(
            epoch=epoch,
            d_loss=sum(D_losses) / len(D_losses),
            g_loss=sum(G_losses) / len(G_losses),
            tpr=tpr,
            tnr=tnr
        )
        print(log)
        print(log, file=log_f)
        
        fake_data_fixed = G(z_fixed)
        image_path = args.output_dir + '/epoch{}.png'.format(epoch)
        square_plot(fake_data_fixed.view(25, 28, 28).cpu().data.numpy(), path=image_path)
        generated_images.append(image_path)
        
        train_hist['D_losses'].append(torch.mean(torch.Tensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.Tensor(G_losses)))

# save

torch.save(G.state_dict(), args.output_dir + 'G.pt')
torch.save(D.state_dict(), args.output_dir + 'D.pt')
with open('loss_history.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

generated_image_array = [imageio.imread(generated_image) for generated_image in generated_images]
imageio.mimsave(args.output_dir + '/GAN_generation.gif', generated_image_array, fps=5)
