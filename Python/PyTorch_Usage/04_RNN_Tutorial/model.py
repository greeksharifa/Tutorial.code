import torch
from torch import nn


class GenderGuesser(nn.Module):

    def __init__(self,
                 input_dim, hidden_dim, output_dim,
                 model='RNN', bias=True, nonlinearity='tanh',
                 use_cuda = False):
        """
        Character RNN which performs speculating gender

        :param input_dim     : same with embedding_dim
        :param hidden_dim    : 1st dimension of hidden layer
        :param output_dim    : same with target dimension
        :param model         : select a kind of RNN model: RNN(default), GRU, LSTM.
        :param bias          : If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param nonlinearity  : nonlinearity – select activation function: tanh(default), relu
        """
        super(GenderGuesser, self).__init__()

        self.input_dim = input_dim   # which is 52(a-zA-Z)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim # which is 2(Male and Female)
        self.model = model
        self.use_cuda = use_cuda

        self.word_embeddings = nn.Embedding(input_dim, hidden_dim)

        # The LSTM takes word embeddings as inputs,
        # and outputs hidden states with dimensionality hidden_dim.
        if model == 'GRU':
            self.rnn = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)
        elif model == 'LSTM':
            self.rnn = nn.LSTMCell(hidden_dim, hidden_dim, bias=bias)
        else: # model == 'RNN'
            self.rnn = nn.RNNCell(hidden_dim, hidden_dim, bias=bias, nonlinearity=nonlinearity)

        # h2o = hidden2output, that maps from hidden state space to output space
        self.h2o = nn.Linear(hidden_dim, output_dim)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hx):
        """
        override method
        :param inputs: tensor of person name, size: [len(name)]
        :param hx: initial hidden layer input
        :return: scores by nn.LogSoftmax
        """
        if self.use_cuda:
            inputs = inputs.cuda()
        inputs = self.word_embeddings(inputs).view(len(inputs), 1, -1) # [len of name, 1, hidden_dim]

        """print('inputs.shape:', inputs.shape)
        print('inputs[i].shape:', inputs[0].shape)
        print('hx.shape:', hx.shape)"""
        # inputs: [1, hidden_dim]
        # hx    : [1, hidden_dim]
        if self.model == 'LSTM':
            hx, cx = hx
            for i in range(len(inputs)):
                hx, cx = self.rnn(inputs[i], (hx, cx))
        else:
            for i in range(len(inputs)):
                hx = self.rnn(inputs[i], hx)

        gender_guessed = self.h2o(hx)
        gender_scores = self.softmax(gender_guessed)

        return gender_scores

    def init_hidden(self):

        hidden = torch.zeros(1, self.hidden_dim)
        if self.use_cuda:
            hidden = hidden.cuda()

        if self.model == 'LSTM':
            return hidden, hidden
        else:
            return hidden

