# dataset.py
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def get_cifar10_dataloader(batch_size=64, image_size=32):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scales to [-1, 1]
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

def generate_random_mask(img_shape, min_size_ratio=0.1, max_size_ratio=0.5):
    """Generates a random rectangular mask."""
    B, C, H, W = img_shape
    mask = torch.ones((B, 1, H, W), dtype=torch.float32) # 1 for known, 0 for unknown

    for i in range(B):
        mask_h = np.random.randint(int(H * min_size_ratio), int(H * max_size_ratio) + 1)
        mask_w = np.random.randint(int(W * min_size_ratio), int(W * max_size_ratio) + 1)
        top = np.random.randint(0, H - mask_h + 1)
        left = np.random.randint(0, W - mask_w + 1)
        mask[i, :, top:top+mask_h, left:left+mask_w] = 0.0
    return mask