
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from model import *
from data_load import load_dataset
from utils import *
from args import get_args
from optim import get_optim

from torch.utils.tensorboard import SummaryWriter

# 主函数
def main():
    '''
    需要处理的超参数
    lr 、batch_size、device、num_epochs、
    model、logdir、seed
 
    '''
    
    #1-实例化argarse
    
    args = get_args()
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
    elif args.model == 'MLP_3':
        model = MLP_3().to(device)
    elif args.model == 'MLP_10':
        model = MLP_10().to(device)
    elif args.model == 'MLP_4X':
        model = get_MLP_4X().to(device)
    elif args.model =='MLP_3s':
        model = get_MLP_3s().to(device)
    elif args.model =='MLP_3t':
        model = get_MLP_3t().to(device)
    elif args.model =='MLP_10_WetInit':
        model = MLP_10_WetInit().to(device)
    else:
        #打印报错信息
        print('model name is not correct')
        
    print(f'args.model：{args.model}')
    print(f'model：{model}')
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)  
    optimizer = get_optim(args.optimizer, model, args.lr, args.momentum)

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
