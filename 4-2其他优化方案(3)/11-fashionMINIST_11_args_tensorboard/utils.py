import torch
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