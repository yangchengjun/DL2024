
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

# 数据集加载与预处理
def load_dataset(batch_size=128):
    train_data = datasets.MNIST(
        root='./data/', train=True, download=True, transform=transforms.ToTensor())
    # subset_indices = list(range(0, 10000))  # 只使用前1万张训练样本
    test_data = datasets.MNIST(
        root='./data/', train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # train_loader = DataLoader(train_data, subset_indices,batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
    
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
# 模型训练
def train_model(model, train_loader, optimizer, criterion, device, writer, epochs):
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 梯度清零
        output = model(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        total_loss += loss.item()
        
        # 计算训练准确率
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f'Epoch {epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epochs} completed. \n Average Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%')
    
    # 将训练误差和准确率写入tensorboard
    writer.add_scalar('training loss', avg_loss, epochs)
    writer.add_scalar('training accuracy', accuracy, epochs)

# 模型测试
def test_model(model, test_loader, criterion, device, writer, epochs):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    print(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}, Epoch: {epochs}')

    # 将测试误差和准确率写入tensorboard
    writer.add_scalar('test loss', avg_loss, epochs)
    writer.add_scalar('test accuracy', accuracy, epochs)

# 主函数
def main():
    '''
    需要处理的超参数
    lr 、batch_size、device、num_epochs、
    model、logdir、seed
 
    '''
    import argparse
    #1-实例化argarse
    args = argparse.ArgumentParser()    #实例化Parser 解释器
    #2-添加参数
    args.add_argument('--batch_size', type=int, default=128,help='num of fig each forward propagation')
    args.add_argument('--lr', type=float, default=0.001,help='learning rate')
    args.add_argument('--num_epochs', type=int, default=50,help='num of epochs')
    args.add_argument('--device', type=str, default='cuda',help="device, 'cuda'  or 'cpu'")
    args.add_argument('--model',type=str,default='MLP_3',help='you can choose 3 or 4 layer MLP,default is MLP_3')
    args.add_argument('--logdir',type=str,default='./logs',help='logdir')
    args.add_argument('--seed',type=int,default=123,help='random seed')
    #3-解析参数
    args = args.parse_args()
    #4-使用参数



    # 设置随机种子
    torch.manual_seed(args.seed)


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = args.device ###############################1
    print(f'Using device: {device}')
    train_loader, test_loader = load_dataset(args.batch_size)   ################2
    # model = MLP().to(device)
    if args.model == 'MLP_4':
        model = MLP_4().to(device)
    elif args.model == 'MLP_10':
        model = MLP_10().to(device)
    elif args.model == 'MLP_4X':
        model = MLP_4X().to(device)
    elif args.model =='MLP_3s':
        model = MLP_3s().to(device)
    elif args.model =='MLP_3t':
        model = MLP_3t().to(device)
    else:
        model = MLP_3().to(device)
    print(f'args.model：{args.model}')
    print(f'model：{model}')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  ################3
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(args.logdir) 
    num_epochs = args.num_epochs    ######################################4
    for epoch in range(num_epochs):
        train_model(model, train_loader, optimizer, criterion, device, writer, epoch)
        test_model(model, test_loader, criterion, device, writer, epoch)
    writer.close()

if __name__ == '__main__':
    main()
