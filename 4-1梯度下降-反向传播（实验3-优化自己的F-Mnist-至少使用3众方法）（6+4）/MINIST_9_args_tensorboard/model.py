import torch
import torch.nn as nn
import torch.nn.functional as F
# 
class MLP_3(nn.Module):
    def __init__(self):
        super(MLP_3, self).__init__()
        self.fc1 = nn.Linear(28*28*1, 1024)
        self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 256)#删除第三层对抗过拟合
        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28*1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))   #删除第三层对抗过拟合
        x = self.fc4(x)  
        return x
    
class MLP_3s(nn.Module):
    def __init__(self):
        super(MLP_3s, self).__init__()
        self.fc1 = nn.Linear(28*28*1, 512)
        self.fc3 = nn.Linear(512, 64)#删除第三层对抗过拟合
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28*1) 
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))   #删除第三层对抗过拟合
        x = self.fc4(x)  
        return x
    
class MLP_3t(nn.Module):
    def __init__(self):
        super(MLP_3t, self).__init__()
        self.fc1 = nn.Linear(28*28*1, 64)
        self.fc3 = nn.Linear(64, 32)#删除第三层对抗过拟合
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28*28*1) 
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))   #删除第三层对抗过拟合
        x = self.fc4(x)  
        return x
    
class MLP_4(nn.Module):
    def __init__(self):
        super(MLP_4, self).__init__()
        self.fc1 = nn.Linear(28*28*1, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)#删除第三层对抗过拟合
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28*1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))   #删除第三层对抗过拟合
        x = self.fc4(x)  
        return x
    
class MLP_4X(nn.Module):
    def __init__(self):
        super(MLP_4X, self).__init__()
        self.fc1 = nn.Linear(28*28*1, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512, 256)#删除第三层对抗过拟合
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28*1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc3(x))   #删除第三层对抗过拟合
        x = self.fc4(x)  
        return x
#写一个10层MLP
class MLP_10(nn.Module):
    def __init__(self):
        super(MLP_10, self).__init__()
        self.fc1 = nn.Linear(28*28*1, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 32)
        self.fc8 = nn.Linear(32, 16)
        self.fc9 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(-1, 28*28*1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x



def get_MLP_3():
    model = MLP_3()
    return model


def get_MLP_3s():
    model = MLP_3s()
    return model

def get_MLP_3t():
    model = MLP_3t()
    return model

def get_MLP_4():
    model = MLP_4()
    return model

def get_MLP_4X():
    model = MLP_4X()
    return model

def get_MLP_10():
    model = MLP_10()
    return model



    
    