import torch
import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
import params

patch_size = 28
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(patch_size, patch_size)),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
    # transforms.Normalize((0.29730626, 0.29918741, 0.27534935),
    #                      (0.32780124, 0.32292358, 0.32056796)),
    # transforms.Normalize((0., 0., 0.),
    #                      (1., 1., 1.)),
])

# root = Root directory of dataset where directory``SVHN`` exists.
root = 'data/pytorch/SVHN'
svhn_train_dataset = datasets.SVHN(root=root, split="train", transform=transform, download=True)
svhn_valid_dataset = datasets.SVHN(root=root, split="train", transform=transform, download=True)
svhn_test__dataset = datasets.SVHN(root=root, split="test", transform=transform, download=True)

indices = list(range(len(svhn_train_dataset)))
validation_size = 5000
train_idx, valid_idx = indices[validation_size:], indices[:validation_size]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

mnist_train_loader = DataLoader(
    svhn_train_dataset,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

mnist_valid_loader = DataLoader(
    svhn_valid_dataset,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

mnist_test_loader = DataLoader(
    svhn_test__dataset,
    batch_size=params.batch_size,
    num_workers=params.num_workers
)
