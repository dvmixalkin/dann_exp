import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
import model.params as params

# import sys
# sys.path.append()


def create_mnist(root='../dann_exp/data/pytorch/MNIST', transform_hyperparameters_version='1'):
    assert transform_hyperparameters_version in ['1', '2', '3'], 'select correct version from [`1`, `2`, `3`]'
    if transform_hyperparameters_version == '1':
        mean = (0.5,)
        std = (0.5,)
    elif transform_hyperparameters_version == '2':
        mean = (0.13092535192648502,)
        std = (0.3084485240270358,)
    else:
        mean = (0.,)
        std = (1.,)
    mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    mnist_train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=mnist_transform)
    mnist_test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=mnist_transform)

    mnist_train_loader = DataLoader(
        mnist_train_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers
    )

    mnist_test_loader = DataLoader(
        mnist_test_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers
    )

    return {'train': mnist_train_loader, 'test': mnist_test_loader}


if __name__ == "__main__":
    loaders = create_mnist()
    print('!')
