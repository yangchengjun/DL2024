import argparse
def get_args():
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
    
    # 增加初始化选项
    args.add_argument('--weight_init',type=str,default='xavier_uniform',help='select init methode in xavier_uniform,xavier_normal,kaiming_uniform,kaiming_normal,uniform,normal')
    
    # 增加优化器选项
    args.add_argument('--optimizer',type=str,default='adam',help='select optimizer in Adam,SGD,RMSprop,proxsgd')
    args.add_argument('--momentum',type=float,default=0.9,help='momentum for SGD')
    
    return  args