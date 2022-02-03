import os
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import SubsetRandomSampler, DataLoader
import model.params as params


# MNIST-M
class CombinedMNIST(data.Dataset):
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file_mnist_m = 'mnist_m_train.pt'
    test_file_mnist_m = 'mnist_m_test.pt'

    def __init__(self, mnist=None, mnistm=None, svhn=None, train=True, transform=None):
        super(CombinedMNIST, self).__init__()

        self.train = train  # training set or test set
        self.transform = transform
        if self.train:
            if mnist:
                mnist_data, mnist_labels = mnist.data, mnist.targets
                mnist_data = torch.stack([mnist_data, mnist_data, mnist_data], 3)
            else:
                mnist_data, mnist_labels = None, None
            if mnistm:
                mnistm_data, mnistm_labels = mnistm.train_data, mnistm.train_labels
            else:
                mnistm_data, mnistm_labels = None, None

            if svhn:
                # svhn_data = torch.as_tensor(svhn.data).permute(0, 2, 3, 1)[:, 2:30, 2:30, :]
                svhn_data = self.resize_svhn(dataloader=svhn, split='train', size=(28, 28))
                svhn_labels = torch.from_numpy(svhn.labels)
            else:
                svhn_data, svhn_labels = None, None

            self.train_data = svhn_data  # torch.cat([mnistm_data, svhn_data])  # mnist_data,
            del svhn_data
            self.train_labels = svhn_labels  # torch.cat([mnistm_labels, ])  # mnist_labels,
            del svhn_labels
        else:
            if mnist:
                mnist_data, mnist_labels = mnist.data, mnist.targets
                mnist_data = torch.stack([mnist_data, mnist_data, mnist_data], 3)
            else:
                mnist_data, mnist_labels = None, None
            if mnistm:
                mnistm_data, mnistm_labels = mnistm.test_data, mnistm.test_labels
            else:
                mnistm_data, mnistm_labels = None, None

            if svhn:
                # svhn_data = torch.as_tensor(svhn.data).permute(0, 2, 3, 1)[:, 2:30, 2:30, :]
                svhn_data = self.resize_svhn(dataloader=svhn, split='test', size=(28, 28))
                svhn_labels = torch.as_tensor(svhn.labels)
            else:
                svhn_data, svhn_labels = None, None

            self.test_data = svhn_data  # torch.cat([mnistm_data, svhn_data])  # mnist_data,
            del svhn_data
            self.test_labels = svhn_labels  # torch.cat([mnistm_labels, svhn_labels])  # mnist_labels,
            del svhn_labels

    @staticmethod
    def resize_svhn(dataloader, split, size: tuple = (28, 28)):
        template = f'./data/pytorch/SVHN'
        npy_path = f'{template}/svhn_{split}.npy'
        image_folder = f'{template}/imgs_{split}'
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        if not os.path.exists(npy_path):
            n, c, w, h = dataloader.data.shape
            npy_to_dump = np.zeros((n, c, size[0], size[1]))
            for idx, (img, lbl) in enumerate(zip(dataloader.data, dataloader.labels)):
                image_path = f'{image_folder}/{idx}_{lbl}.png'
                resized_img = cv2.resize(img.transpose(1, 2, 0), (28, 28))
                Image.fromarray(resized_img).save(image_path)
                npy_to_dump[idx] = resized_img.transpose(2, 0, 1)
            np.save(file=npy_path, arr=npy_to_dump)
            svhn_data = torch.from_numpy(npy_to_dump)
        else:
            svhn_data = torch.from_numpy(np.load(npy_path))
        return svhn_data.permute(0, 2, 3, 1)

    def __getitem__(self, index):

        """Get images and target for data loader.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(type(img))
        img = Image.fromarray(img.squeeze().numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


mnist_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5,),
                                                           (0.5,))
                                      # transforms.Normalize((0.13092535192648502,),
                                      #                      (0.3084485240270358,))
                                      # transforms.Normalize((0.,),
                                      #                      (1.,))
                                      ])
mnist_train_dataset = datasets.MNIST(root='data/pytorch/MNIST',
                                     train=True,
                                     download=True,
                                     transform=mnist_transform)
mnist_test_dataset = datasets.MNIST(root='data/pytorch/MNIST',
                                    train=False,
                                    download=True,
                                    transform=mnist_transform)
mnist = {'train': mnist_train_dataset, 'test': mnist_test_dataset}

# MNIST_M
from datasets.mnistm import MNISTM

mnistm_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.29730626, 0.29918741, 0.27534935),
                                                            (0.32780124, 0.32292358, 0.32056796))])

mnistm_train_dataset = MNISTM(root='data/pytorch/MNIST-M',
                              train=True,
                              download=True,
                              transform=mnistm_transform)
mnistm_test_dataset = MNISTM(root='data/pytorch/MNIST-M',
                             train=False,
                             download=True,
                             transform=mnistm_transform)
mnistm = {'train': mnistm_train_dataset, 'test': mnistm_test_dataset}

# SVHN
patch_size = 28
transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(28),
    transforms.ToTensor(),
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
svhn_test__dataset = datasets.SVHN(root=root, split="test", transform=transform, download=True)
svhn = {'train': svhn_train_dataset, 'test': svhn_test__dataset}

# COMBINED DATASET
transform = transforms.Compose([transforms.ToTensor()])
combined_train_dataset = CombinedMNIST(mnist['train'], mnistm['train'], svhn['train'], train=True, transform=transform)
combined_valid_dataset = CombinedMNIST(mnist['train'], mnistm['train'], svhn['train'], train=True, transform=transform)
combined_test_dataset = CombinedMNIST(mnist['test'], mnistm['test'], svhn['test'], train=False, transform=transform)

indices = list(range(len(combined_train_dataset)))
validation_size = 10000
train_idx, valid_idx = indices[validation_size:], indices[:validation_size]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

combined_train_loader = DataLoader(
    combined_train_dataset,
    batch_size=params.batch_size,
    sampler=train_sampler,
    num_workers=params.num_workers
)

combined_valid_loader = DataLoader(
    combined_valid_dataset,
    batch_size=params.batch_size,
    sampler=valid_sampler,
    num_workers=params.num_workers
)

combined_test_loader = DataLoader(
    combined_test_dataset,
    batch_size=params.batch_size,
    num_workers=params.num_workers
)
