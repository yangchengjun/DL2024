
'''
4L-Epoch 9 completed. 
Training Accuracy: 75.49% Average Loss: 0.6794, 
Test Accuracy: 54.85%,       Test Loss: 1.5971
over fitting


3L:
 Average Loss: 0.6673, Training Accuracy: 76.28%
Test Accuracy: 53.30%, Test Loss: 1.6377, Epoch: 9
'''


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
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])

    train_data = datasets.CIFAR10(
        root='./data/cifar10', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(
        root='./data/cifar10', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 256)#删除第三层对抗过拟合
        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))   #删除第三层对抗过拟合
        x = self.fc4(x)  
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    train_loader, test_loader = load_dataset()
    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter('runs/LOG_CIFAR10_3L')
    num_epochs = 10
    for epoch in range(num_epochs):
        train_model(model, train_loader, optimizer, criterion, device, writer, epoch)
        test_model(model, test_loader, criterion, device, writer, epoch)
    writer.close()

if __name__ == '__main__':
    main()
