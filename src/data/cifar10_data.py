"""

For loading CIFAR-10 data


"""

import pickle

import torch
import torchvision
import torchvision.transforms as transforms



def load_cifar10(batch_size: int, shuffle_train: bool):
    """ load CIFAR-10 data """
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./downloaded_data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=shuffle_train, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./downloaded_data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    file = './downloaded_data/cifar-10-batches-py/batches.meta'
    with open(file, 'rb') as fo:
        meta = pickle.load(fo, encoding='bytes')
    label_names = [bs.decode('latin1') for bs in meta[b'label_names']]
    
    return trainloader, testloader, label_names
