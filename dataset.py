import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import Grayscale
import os


# def grayscale_to_3d(image):
#     gray = np.dot(image[...,:3], [0.333, 0.333, 0.334])
#     gray_3d = np.expand_dims(gray, axis=3)
#     return gray_3d



# 1*28*28
def Fashion_MNIST_dataset(batch_size, test_batch_size, into_grey = False):
    
    train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=test_batch_size)
    
    return train_set, test_set, train_loader, test_loader

# 1*28*28
def MNIST_dataset(batch_size, test_batch_size, into_grey = False):
    
    train_set = torchvision.datasets.MNIST("./data", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.MNIST("./data", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=test_batch_size)
    
    return train_set, test_set, train_loader, test_loader

# 3*32*32
def Cifar_10_dataset(batch_size, test_batch_size, into_grey = False):
    if into_grey:
        transform = transforms.Compose([transforms.Resize(28),
                                        transforms.Grayscale(),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.5,), (0.5,))
                                        ])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

    train_set = datasets.CIFAR10('./data/cifar10', train=True,download=True,
                                                                transform=transform)

    test_set = datasets.CIFAR10('./datasets/cifar10', train=False,download=True, 
                                                              transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    val_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size)
    
   
    return train_set, test_set, train_loader, val_loader


# 3*32*32
def SVHN_dataset(batch_size, test_batch_size, into_grey = False):
    if into_grey:
        transform = transforms.Compose([transforms.Resize(28),
                                        transforms.Grayscale(),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.5,), (0.5,))
                                        ])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        # transforms.Normalize((0.5,), (0.5,))
                                        ])
    
    train_set = datasets.SVHN('./data/svhn/', split='train',transform=transform,
                                                         download=True)
    test_set = datasets.SVHN('./data/svhn/', split='test', transform=transform, 
                                                       download=True)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_set,
                                             batch_size=test_batch_size, shuffle=True)
    
    return train_set, test_set, train_loader, val_loader

# 3*64*64
def TinyImagenet_r_dataset(batch_size, test_batch_size, into_grey = False):
    if into_grey:
        transform = transforms.Compose([transforms.Resize(28),
                                        transforms.Grayscale(),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.5,), (0.5,))
                                        ])
    else:
        transform = transforms.Compose([transforms.Resize(32),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.5,), (0.5,))
                                        ])
    
    train_datasets = datasets.ImageFolder(os.path.join('./data/tiny-imagenet-200', 'train'), transform=transform) 
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    
    test_datasets = datasets.ImageFolder(os.path.join('./data/tiny-imagenet-200', 'test'), transform=transform) 
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=test_batch_size, shuffle=True)
    
    return train_datasets, test_datasets, train_loader, test_loader

# 3*64*64
def TinyImagenet_c_dataset(batch_size, test_batch_size, into_grey = False):
    if into_grey:
        transform = transforms.Compose([transforms.Resize(28),
                                        transforms.Grayscale(),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.5,), (0.5,))
                                        ])
    else:
        transform = transforms.Compose([transforms.Resize(32),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.5,), (0.5,))
                                        ])
    
    train_datasets = datasets.ImageFolder(os.path.join('./data/tiny-imagenet-200', 'train'), transform=transform) 
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    
    test_datasets = datasets.ImageFolder(os.path.join('./data/tiny-imagenet-200', 'test'), transform=transform) 
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=test_batch_size, shuffle=True)
    
    return train_datasets, test_datasets, train_loader, test_loader
    
    
    
if __name__ == "__main__":   
    a_m, b_m, c_m, d_m = MNIST_dataset(batch_size = 64, test_batch_size = 64)
    a_c, b_c, c_c, d_c = Cifar_10_dataset(batch_size = 64, test_batch_size = 64, into_grey = True)
    # print(a)
    print(b_m[0][0].size())
    print(b_c[0][0].size())

    # a.data = grayscale_to_3d(a.data)
    # print(a.data.shape)

    i = 0
    for data, target in d_m:
        print(data.shape)
        if i == 0: 
            break

    for data, target in d_c:
        print(data.shape)
        if i == 0: 
            break