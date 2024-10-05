from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 数据集加载与预处理
def load_dataset(batch_size=128):
    train_data = datasets.FashionMNIST(
        root='../data/', train=True, download=True, transform=transforms.ToTensor())
    # subset_indices = list(range(0, 10000))  # 只使用前1万张训练样本
    test_data = datasets.FashionMNIST(
        root='../data/', train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # train_loader = DataLoader(train_data, subset_indices,batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader