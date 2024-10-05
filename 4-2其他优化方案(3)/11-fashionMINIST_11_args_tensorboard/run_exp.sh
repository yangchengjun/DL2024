# 注意第一行不会识别
# !/bin/bash 
# 对比不同初始化方法
# python fashionMNIST_11_args_tensorboard_early.py --mod MLP_3 --logdir './logs/MINIST/MLP_3'
# python fashionMNIST_11_args_tensorboard_early.py --mod MLP_10 --logdir './logs/MINIST/MLP_10'
# python fashionMNIST_11_args_tensorboard_early.py --mod MLP_10_WetInit --logdir './logs/MINIST/MLP_10_WetInit'
# python fashionMNIST_11_args_tensorboard_early.py --mod MLP_10_WetInit --logdir './logs/MINIST/MLP_10_WetInit_bias0'


# 对比不同优化器
python fashionMNIST_11_args_tensorboard_early.py --mod MLP_3 --use_early_stop ture --logdir './logs/fashionMINIST_optim/MLP_3_SGD' --optimizer 'sgd'
python fashionMNIST_11_args_tensorboard_early.py --mod MLP_3 --use_early_stop ture --logdir './logs/fashionMINIST_optim/MLP_3_adam' --optimizer 'adam'
python fashionMNIST_11_args_tensorboard_early.py --mod MLP_3 --use_early_stop ture --logdir './logs/fashionMINIST_optim/MLP_3_RMSprop' --optimizer 'RMSprop'





