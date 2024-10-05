
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import *

from torch.utils.tensorboard import SummaryWriter

#############################################################################
def early_stopping(val_loss, best_val_loss, patience_counter, early_stop_patience, early_stop_delta):
    if val_loss < best_val_loss - early_stop_delta:
        best_val_loss = val_loss
        patience_counter = 0  # 重置耐心计数器
    else:
        patience_counter += 1
    # 判断是否需要早停
    stop_training = patience_counter >= early_stop_patience
    return best_val_loss, patience_counter, stop_training
#####################################################


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
    
    # 3-test_model 函数以返回平均损失：################################################################
    return avg_loss

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
    args.add_argument('--logdir',type=str,default='./logs/test',help='logdir')
    args.add_argument('--seed',type=int,default=123,help='random seed')
    
    # 增加早停法相关参数
    args.add_argument('--use_early_stop',type=bool,default=False,help='whether to use early stop Ture or False')
    args.add_argument('--early_stop_patience', type=int, default=5, help='number of epochs with no improvement after which training will be stopped')
    args.add_argument('--early_stop_delta', type=float, default=0.001, help='minimum change in the monitored quantity to qualify as an improvement')
    #3-解析参数
    args = args.parse_args()
    #4-使用参数

    # 设置随机种子
    torch.manual_seed(args.seed)
    # 早停相关变量############################################################################
    best_val_loss = float('inf')
    patience_counter = 0   
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = args.device 
    print(f'Using device: {device}')
    train_loader, test_loader = load_dataset(args.batch_size)   
    # model = MLP().to(device)
    if args.model == 'MLP_4':
        model = get_MLP_4().to(device)
    elif args.model == 'MLP_10':
        model = MLP_10().to(device)
    elif args.model == 'MLP_4X':
        model = get_MLP_4X().to(device)
    elif args.model =='MLP_3s':
        model = get_MLP_3s().to(device)
    elif args.model =='MLP_3t':
        model = get_MLP_3t().to(device)
    else:
        model = get_MLP_3().to(device)
        
    print(f'args.model：{args.model}')
    print(f'model：{model}')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(args.logdir) 
    num_epochs = args.num_epochs 
    for epoch in range(num_epochs):
        train_model(model, train_loader, optimizer, criterion, device, writer, epoch)
        # test_model(model, test_loader, criterion, device, writer, epoch)###############


        val_loss = test_model(model, test_loader, criterion, device, writer, epoch)

        # 执行早停法逻辑###########################################################
        if args.use_early_stop:
            print('use early stop')
            best_val_loss, patience_counter, stop_training = early_stopping(
                val_loss, best_val_loss, patience_counter, args.early_stop_patience, args.early_stop_delta
            )
            if stop_training:
                print(f'Early stopping at epoch :{epoch}')
                print(f'patience_counter:{patience_counter}')
                print(f'best_VAl_loss:{best_val_loss}')
                break
        else:
            print('disable early stop')
###################################################################################
    writer.close()

if __name__ == '__main__':
    main()
