import torch
import torch.nn as nn
import torch.nn.functional as F


class Ssl_model_rot(nn.Module): # after the avg pooling layer
    def __init__(self, indim):
        super(Ssl_model_rot, self).__init__()

        self.bn3 = nn.BatchNorm1d(512)
        self.linear = nn.Linear(indim, 512)
        self.linear2 = nn.Linear(512, 4)
        

    def forward(self, x):
        x = F.relu(self.bn3(self.linear(x)))
        x = self.linear2(x)
        return x


class Ssl_model_contrast(nn.Module): # after the avg pooling layer
    def __init__(self, default_in_dim=640): #2048
        super(Ssl_model_contrast, self).__init__()

        self.bn3 = nn.BatchNorm1d(256) #512
        self.linear = nn.Linear(default_in_dim, 256) #512
        self.linear2 = nn.Linear(256, 128) #512, 128

    def forward(self, x):
        x = F.relu(self.bn3(self.linear(x)))
        x = self.linear2(x)
        
        #return x
        return  x


class Ssl_model_contrast2(nn.Module): # after the avg pooling layer
    def __init__(self, default_in_dim=640): #2048
        super(Ssl_model_contrast2, self).__init__()

        self.bn3 = nn.BatchNorm1d(512) #512
        self.linear = nn.Linear(default_in_dim, 512) #512
        self.linear2 = nn.Linear(512, 128) #512, 128

    def forward(self, x):
        x = F.relu(self.bn3(self.linear(x)))
        x = self.linear2(x)
        
        #return x
        return  x

class rot_out_branch(nn.Module): # after the avg pooling layer
    def __init__(self, default_in_dim=640):
        super(rot_out_branch, self).__init__()

        self.bn3 = nn.BatchNorm1d(256)
        self.linear = nn.Linear(default_in_dim, 256)
        self.linear2 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.bn3(self.linear(x)))
        x = self.linear2(x)
        return x