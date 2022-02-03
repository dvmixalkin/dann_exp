import torchvision.datasets as datasets
from torchvision import transforms
import model.params as params
from torch.utils.data import DataLoader


def create_svhn(root='../dann_exp/data/pytorch/SVHN', patch_size=28, transform_hyperparameters_version='1'):
    assert transform_hyperparameters_version in ['1', '2', '3'], 'select correct version from [`1`, `2`, `3`]'
    if transform_hyperparameters_version == '1':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif transform_hyperparameters_version == '2':
        mean = (0.29730626, 0.29918741, 0.27534935)
        std = (0.32780124, 0.32292358, 0.32056796)
    else:
        mean = (0., 0., 0.)
        std = (1., 1., 1.)

    transform = transforms.Compose([
        transforms.Resize(patch_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    svhn_train_dataset = datasets.SVHN(root=root, split="train", transform=transform, download=True)
    svhn_test__dataset = datasets.SVHN(root=root, split="test", transform=transform, download=True)

    mnist_train_loader = DataLoader(
        svhn_train_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers
    )

    mnist_test_loader = DataLoader(
        svhn_test__dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers
    )
    return {'train': mnist_train_loader, 'test': mnist_test_loader}
