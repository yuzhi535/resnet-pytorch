from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
# import albumentations as A
import numpy as np
from torchvision import transforms

# from sklearn.model_selection import train_test_split


def get_CIFAdataset_loader(root, batch_size, num_workers, pin_memory, valid_rate, shuffle: bool, random_seed=42, augment=True):
    # 引入分割CIFA数据集的包，分割数据为训练集和验证集
    from torch.utils.data.sampler import SubsetRandomSampler

    # 预处理
    train_transform = transforms.Compose([
        transforms.Compose([transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(), ]) if augment else transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
    ]
    )

    val_transform = transforms.Compose([
        transforms.Compose([transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(), ]) if augment else transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
    ]
    )

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
    ]
    )

    # CIFA-10数据集
    train_dataset = CIFAR10(root=root, train=True,
                            download=True, transform=train_transform)
    val_dataset = CIFAR10(root=root, train=True,
                          download=False, transform=val_transform)
    test_dataset = CIFAR10(root=root, train=False,
                           download=True, transform=test_transform)

    # 分割数据集
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_rate * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # 生成dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=1,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader

