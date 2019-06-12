import argparse
import os

from torch import nn
from torch import optim

from model import GenderGuesser
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="RNN", choices=['RNN', 'GRU', 'LSTM'], help='select a kind of RNN model: RNN, GRU, LSTM.')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--hidden_size', type=int, default=64, help='1st dimension of hidden layer')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate of optimizer')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of input of model')
parser.add_argument('--bias', type=bool, default=True, help='set bias of model.rnn')
parser.add_argument('--nonlinearity', type=bool, default=False, help='set activation function of model.rnn')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

os.makedirs('models/', exist_ok=True)

training_set = read_files()

n_letters, all_letters = get_all_letters()
print(n_letters, all_letters)

word_to_idx = {letter: idx for (idx, letter) in enumerate(all_letters)}
gender_to_idx = {'M': 0, 'F': 1}

hidden_dim = 128


model = GenderGuesser(n_letters, hidden_dim, 2,
                      model=args.model, bias=args.bias, nonlinearity='tanh',
                      use_cuda=use_cuda)

if use_cuda:
    model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)

print(training_set[:5])

loss_list = []

for epoch in range(1, args.epochs + 1):
    print('\repoch: {}'.format(epoch))
    total_loss = 0.0
    loss_1000 = 0.0

    random.shuffle(training_set)

    for i, (name, gender) in enumerate(training_set):
        # print(name, gender)
        # print(name, gender)
        model.zero_grad()

        # nameTensor = name_to_tensor(name)
        # gender_codeTensor = gender_code_to_tensor(gender_code)
        nameTensor = prepare_sequence(name, word_to_idx)
        genderTensor = prepare_sequence(gender, gender_to_idx)
        if use_cuda:
            nameTensor = nameTensor.cuda()
            genderTensor = genderTensor.cuda()

        hidden = model.init_hidden()
        gender_scores = model(nameTensor, hidden)

        loss = criterion(gender_scores, genderTensor)
        loss_1000 += loss
        total_loss += loss

        if (i+1) % 1000 == 0:
            print('idx {:6d} | name: {:<15s} | gender: {} | loss={:.6f}'.format(i+1, name, gender, loss_1000))
            loss_1000 = 0.0

        loss.backward()
        optimizer.step()

    loss_list.append(total_loss)
    print(loss_list); print(loss_list, file=open('log.txt', 'w', encoding='utf-8'))

    torch.save(model.state_dict(), 'models/model={}_epoch={:03d}.pt'.format(args.model, epoch))

print('\n\mEvaluate:')



test_names = ['Mary', 'Anna', 'Emma', 'John', 'William', 'James']


for name in test_names:
    nameTensor = prepare_sequence(name, word_to_idx)
    if use_cuda:
        nameTensor.cuda()

    output = model(nameTensor)
    print('name: {:8s}, output: {}'.format(name, output))
