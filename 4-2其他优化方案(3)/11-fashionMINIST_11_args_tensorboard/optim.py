# from traceback import print_tb
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
import math


def get_optim(optimizer, model, lr, momentum):
    if optimizer == 'adam':
        return optim.Adam(
            [param for param in model.parameters() if param.requires_grad], # 只优化需要梯度的参数
            lr=lr,
            weight_decay=5e-4   # 权重衰减
        )
    elif optimizer == 'sgd':
        return optim.SGD(
            [param for param in model.parameters() if param.requires_grad], # 只优化需要梯度的参数
            lr=lr,
            momentum=momentum,
            weight_decay=5e-4   # 权重衰减
        )
    
    elif optimizer == 'RMSprop':
        return optim.RMSprop(
            [param for param in model.parameters() if param.requires_grad], # 只优化需要梯度的参数
            lr=lr,
            momentum=momentum,
            weight_decay=5e-4   # 权重衰减
        )
    
    else:
        raise ValueError('Optimizer not recognized')



    r"""
      Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used (default: None)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,weight_decay=0,curi=0, nesterov=False):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, curi=curi,nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LMSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            weight_decay=group['weight_decay']
            momentum=group['momentum']
            lr=group['lr']
            dampening=group['dampening']
            nesterov=group['nesterov']
            curi=group['curi']
            

           

            for p in group['params']:
                if p.grad is not None:
                    d_p = p.grad

                    state = self.state[p]
                    rare = state['y_sample_size_min']
                    rare_momentum = math.atan(1/rare)*curi
                    # rare_momentum = math.atan(curi/rare)
                    
                    if 'momentum_buffer' not in state:
                        momentum_buffer = None
                    else:
                        momentum_buffer = state['momentum_buffer']

                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)

                    if momentum != 0:
                        buf = momentum_buffer

                        if buf is None:
                            buf = torch.clone(d_p).detach()
                            state['momentum_buffer']= buf
                        else:
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)#.add_(d_p*(rare_momentum),alpha=1-dampening)

                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf

                    # alpha = lr if maximize else -lr
                    # p.data.add_(d_p, alpha=-lr)
                    if rare_momentum == 0:
                        p.data.add_(d_p, alpha=-lr)
                    else:
                        p.data.add_(d_p, alpha=-lr*rare_momentum)
        
        return loss

    def total_y_value(self, y):
        y_classes_gruops = dict()
        now_y_classes = dict()

        y_classes_param_total = list()
        now_y_classes_param = list()

        if len(y) == 0:
            raise ValueError("optimizer got an empty parameter list")

        list_y = [i for i in y]  # 将 dtype 为 tensor 的转为 numpy 类型

        # print("list_y:".format([i for i in list(list_y)]))
          # 构造一个字典
        # y_classes_gruops_list={'y_classes_gruops':{}}
        # 将 'y_classes_gruops' 字典置在 网络接结构中的最顶层，目的减少 ‘y_classes_gruops'的次数
        # y_classes_gruops_flage = True
        for param_group in self.param_groups:

            #print("defore_param_group:{}".format(param_group))
            if isinstance(param_group,dict):
                if 'y_classes_gruops' in param_group.keys():
                    y_classes_gruops = param_group['y_classes_gruops'] # 获取历史batch y 类别和统计数值

          # 获取 在 list_y 内出现的类别，并统计
        for y_name in list_y:
            flag = True

            for key in now_y_classes.keys():
                
                if y_name == key:
                    now_y_classes[key] = now_y_classes[key] + 1
                    flag = False
                    break
            if flag:
                now_y_classes[y_name] = 1

        # 获取 在 list_y 内出现的类别，并和历史出现的进行统计
        for y_name in list_y:
            flag = True
            for key in y_classes_gruops.keys():       
                if y_name == key:
                    y_classes_gruops[key] = y_classes_gruops[key] + 1
                    flag = False
                    break
            if flag:
                y_classes_gruops[y_name] = 1


        # print("y_classes_gruops:{},now_y_classes:{}".format(y_classes_gruops, now_y_classes))

        # now_y_classes_param 获取  在这个 batc size 内含有 y 类别的出现数量，仅含在这个 batch size 内
        #  y_classes_param_total 获取 在这个 batch size 内有含有 y 类别的出现数量 ，含这个 batch size 和之前的

        for y_name in now_y_classes.keys():

            for key in y_classes_gruops.keys():
                if y_name == key:
                    now_y_classes_param.append(y_classes_gruops[key])  #将这一个batch y 类别历史统计的大小存入
                    break

        for y_name in y_classes_gruops.keys():

            y_classes_param_total.append(y_classes_gruops[y_name]) #将所有的类别统计的大小存入


        y_classes_param_total = torch.tensor(y_classes_param_total,
                                             dtype=torch.float)  # 将 dtype: numpy 转变为 tensor float 类型
        now_y_classes_param = torch.tensor(now_y_classes_param, dtype=torch.float)

        y_classes_sum = y_classes_param_total.sum()  # 求类别样本和
        # print("y_classes_sum:{}".format(y_classes_sum))

        y_classes_sample_size = now_y_classes_param.div(y_classes_sum)

        # print("y_classes_sample_size:{}".format(y_classes_sample_size))
        # 计算这一个batch类别样本占比的平均值

        y_sample_size_min = y_classes_sample_size.min(0)

        # print("y_sample_size_min:{}".format(y_sample_size_min[0]))

        for param_group in self.param_groups:
            for param in param_group['params']:
                param_state = self.state[param]
                param_state['y_sample_size_min'] = torch.clone(y_sample_size_min[0].data)  # 将当前batch y 类别的最小稀有度的值加入到 模型的参数里
              
        for param_group in self.param_groups:

            if isinstance(param_group,dict):
                # if 'y_classes_gruops' in param_group.keys():
                param_group['y_classes_gruops'] = y_classes_gruops