import os
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
from torch.utils.data import DataLoader
import model.params as params
from datasets.mnist import create_mnist
from datasets.mnistm import create_mnist_m
from datasets.svhn import create_svhn


class CombinedMNIST(data.Dataset):
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file_mnist_m = 'mnist_m_train.pt'
    test_file_mnist_m = 'mnist_m_test.pt'

    def __init__(self, ds_template, train=True, transform=None):
        super(CombinedMNIST, self).__init__()

        self.train = train  # training set or test set
        self.transform = transform

        mnist = ds_template['mnist'] if ds_template['mnist'] else None
        mnistm = ds_template['mnistm'] if ds_template['mnistm'] else None
        svhn = ds_template['svhn'] if ds_template['svhn'] else None

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

            self.train_data = torch.cat([mnist_data, mnistm_data, svhn_data])
            del svhn_data
            self.train_labels = torch.cat([mnist_labels, mnistm_labels, svhn_labels])
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

            self.test_data = torch.cat([mnist_data, mnistm_data, svhn_data])
            del svhn_data
            self.test_labels = torch.cat([mnist_labels, mnistm_labels, svhn_labels])
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


def create_loaders(datasets_list=['mnist', 'mnist_m', 'svhn'], transform_params=['1', '1', '1']):
    assert len(datasets_list)==len(transform_params), 'set correct sequences for datasets and transform params'
    if isinstance(transform_params, str):
        parameters = [transform_params, transform_params, transform_params]

    mnist = create_mnist(transform_hyperparameters_version=parameters[0]) if 'mnist' in datasets_list else None
    mnistm = create_mnist_m(transform_hyperparameters_version=parameters[1]) if 'mnist_m' in datasets_list else None
    svhn = create_svhn(transform_hyperparameters_version=parameters[2]) if 'svhn' in datasets_list else None
    datasets_template = {
        'mnist': mnist, 'mnistm': mnistm, 'svhn': svhn
    }
    transform = None  # transforms.Compose([transforms.ToTensor()])

    combined_train_dataset = CombinedMNIST(datasets_template, train=True, transform=transform)
    combined_test_dataset = CombinedMNIST(datasets_template, train=False, transform=transform)

    combined_train_loader = DataLoader(
        combined_train_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers
    )

    combined_test_loader = DataLoader(
        combined_test_dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers
    )
    return combined_train_loader, combined_test_loader
