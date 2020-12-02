# -*- : coding: utf-8 -*

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# unsupervised deep learning
class VAE(nn.Module):
    def __init__(self, xn, yn):
        super(VAE, self).__init__()
        # encoder layers
        self.inSize = xn * yn
        self.fc1 = nn.Linear(self.inSize, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc51 = nn.Linear(16,5)
        self.fc52 = nn.Linear(16,5)
        #self.fc6 = nn.Linear(8,4)
        #self.fc71 = nn.Linear(4, 2)
        #self.fc72 = nn.Linear(4, 2)
        # decoder layers
        #self.fc8 = nn.Linear(2, 4)
        #self.fc9 = nn.Linear(4,8)
        self.fc6 = nn.Linear(5,16)
        self.fc7 = nn.Linear(16,32)
        self.fc8 = nn.Linear(32, 64)
        self.fc9 = nn.Linear(64, 128)
        self.fc10 = nn.Linear(128, self.inSize)
    
    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) 
        x = F.relu(self.fc4(x)) 
        #x = F.relu(self.fc5(x)) 
        #x = F.relu(self.fc6(x))
        return (self.fc51(x), self.fc52(x))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = F.relu(self.fc6(z))
        z = F.relu(self.fc7(z))
        z = F.relu(self.fc8(z))
        z = F.relu(self.fc9(z))
        #z = F.relu(self.fc12(z))
        #z = F.relu(self.fc13(z))
        z = self.fc10(z)
        return (z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        encoded = self.reparametrize(mu, logvar)
        decoded = self.decode(encoded)
        label = torch.softmax(encoded, dim = 1)
        return (encoded, decoded, label)