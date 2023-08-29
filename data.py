import torch
from torchvision.datasets import CIFAR10, MNIST, SVHN, FashionMNIST, CIFAR100
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_file_from_google_drive
from utils import DataAugmentation

transform_color = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_gray = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

dino_transform = DataAugmentation(size=224, n_local_crops=2)

def get_train_dataset(dataset, label_class, path, backbone):
    if dataset == "cifar10":
        return get_CIFAR10_train(label_class, path, backbone)
    elif dataset == "cifar100":
        return get_CIFAR100_train(label_class, path, backbone)
    elif dataset == "mnist":
        return get_MNIST_train(label_class, path, backbone)
    elif dataset == "fashion":
        return get_FASHION_MNIST_train(label_class, path, backbone)
    elif dataset == "svhn":
        return get_SVHN_train(label_class, path, backbone)
    elif dataset == "mvtec":
        return get_MVTEC_train(label_class, path, backbone)
    elif dataset == "cub":
        return get_CUB_train(label_class, path, backbone)
    else:
        raise Exception("Source Dataset is not supported yet. ")
        exit()


def get_test_dataset(dataset, normal_labels, path, backbone):
    if dataset == "cifar10":
        return get_CIFAR10_test(normal_labels, path, backbone)
    elif dataset == "cifar100":
        return get_CIFAR100_test(normal_labels, path, backbone)
    elif dataset == "mnist":
        return get_MNIST_test(normal_labels, path, backbone)
    elif dataset == "fashion":
        return get_FASHION_MNIST_test(normal_labels, path, backbone)
    elif dataset == "svhn":
        return get_SVHN_test(normal_labels, path, backbone)
    elif dataset == "mvtec":
        return get_MVTEC_test(normal_labels, path, backbone)
    elif dataset == "cub":
        return get_CUB_test(normal_labels, path, backbone)
    else:
        raise Exception("Target Dataset is not supported yet. ")
        exit()


def get_CIFAR10_train(normal_class_labels, path, backbone):
    transform = transform_color

    trainset = CIFAR10(root=path, train=True, download=True, transform=transform)

    normal_mask = np.isin(trainset.targets, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.targets = [0 for _ in trainset.targets]

    trainset_moco = CIFAR10(root=path, train=True, download=True, transform=dino_transform)

    normal_mask = np.isin(trainset_moco.targets, normal_class_labels)

    trainset_moco.data = trainset_moco.data[normal_mask]
    trainset_moco.targets = [0 for _ in trainset_moco.targets]

    return trainset, trainset_moco


def get_CIFAR10_test(normal_class_labels, path, backbone):
    transform = transform_color

    testset = CIFAR10(root=path, train=False, download=True, transform=transform)
    test_mask = np.isin(testset.targets, normal_class_labels)

    testset.targets = np.array(testset.targets)
    testset.targets[test_mask] = 0
    testset.targets[~test_mask] = 1

    return testset


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


def get_CIFAR100_train(normal_class_labels, path, backbone):
    transform = transform_color

    trainset = CIFAR100(root=path, train=True, download=True, transform=transform)
    trainset.targets = sparse2coarse(trainset.targets)

    normal_mask = np.isin(trainset.targets, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.targets = [0 for _ in trainset.targets]

    trainset_moco = CIFAR100(
        root=path, train=True, download=True, transform=dino_transform
    )
    trainset_moco.targets = sparse2coarse(trainset_moco.targets)

    normal_mask = np.isin(trainset_moco.targets, normal_class_labels)

    trainset_moco.data = trainset_moco.data[normal_mask]
    trainset_moco.targets = [0 for _ in trainset_moco.targets]

    return trainset, trainset_moco


def get_CIFAR100_test(normal_class_labels, path, backbone):
    transform = transform_color

    testset = CIFAR100(root=path, train=False, download=True, transform=transform)
    testset.targets = sparse2coarse(testset.targets)

    test_mask = np.isin(testset.targets, normal_class_labels)

    testset.targets = np.array(testset.targets)
    testset.targets[test_mask] = 0
    testset.targets[~test_mask] = 1

    return testset


def get_MNIST_train(normal_class_labels, path, backbone):
    transform = transform_gray

    trainset = MNIST(root=path, train=True, download=True, transform=transform)

    normal_mask = np.isin(trainset.targets, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.targets = [0 for _ in trainset.targets]

    trainset_moco = MNIST(
        root=path, train=True, download=True, transform=dino_transform
    )

    normal_mask = np.isin(trainset_moco.targets, normal_class_labels)

    trainset_moco.data = trainset_moco.data[normal_mask]
    trainset_moco.targets = [0 for _ in trainset_moco.targets]

    return trainset, trainset_moco


def get_MNIST_test(normal_class_labels, path, backbone):
    transform = transform_gray

    testset = MNIST(root=path, train=False, download=True, transform=transform)
    test_mask = np.isin(testset.targets, normal_class_labels)

    testset.targets = np.array(testset.targets)
    testset.targets[test_mask] = 0
    testset.targets[~test_mask] = 1

    return testset


def get_FASHION_MNIST_train(normal_class_labels, path, backbone):
    transform = transform_gray

    trainset = FashionMNIST(root=path, train=True, download=True, transform=transform)

    normal_mask = np.isin(trainset.targets, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.targets = [0 for _ in trainset.targets]

    trainset_moco = FashionMNIST(
        root=path, train=True, download=True, transform=dino_transform
    )

    normal_mask = np.isin(trainset_moco.targets, normal_class_labels)

    trainset_moco.data = trainset_moco.data[normal_mask]
    trainset_moco.targets = [0 for _ in trainset_moco.targets]

    return trainset, trainset_moco


def get_FASHION_MNIST_test(normal_class_labels, path, backbone):
    transform = transform_gray 

    testset = FashionMNIST(root=path, train=False, download=True, transform=transform)

    test_mask = np.isin(testset.targets, normal_class_labels)

    testset.targets = np.array(testset.targets)
    testset.targets[test_mask] = 0
    testset.targets[~test_mask] = 1

    return testset


def get_SVHN_train(normal_class_labels, path, backbone):
    transform = transform_color

    trainset = SVHN(root=path, split="train", download=True, transform=transform)

    normal_mask = np.isin(trainset.labels, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.labels = [0 for _ in trainset.labels]

    trainset_moco = SVHN(root=path, split="train", download=True, transform=dino_transform)

    normal_mask = np.isin(trainset_moco.labels, normal_class_labels)

    trainset_moco.data = trainset_moco.data[normal_mask]
    trainset_moco.labels = [0 for _ in trainset_moco.labels]

    return trainset, trainset_moco


def get_SVHN_test(normal_class_labels, path, backbone):
    transform = transform_color

    testset = SVHN(root=path, split="test", download=True, transform=transform)
    test_mask = np.isin(testset.labels, normal_class_labels)

    testset.labels = np.array(testset.labels)
    testset.labels[test_mask] = 0
    testset.labels[~test_mask] = 1

    return testset


from tqdm import tqdm
import requests
import subprocess
import os
from glob import glob
from PIL import Image
from torch.utils.data import ConcatDataset

mvtec_labels = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        category,
        transform=None,
        target_transform=None,
        train=True,
        normal=True,
        download=False,
    ):
        self.transform = transform

        # Check if dataset directory exists
        dataset_dir = os.path.join(root, "mvtec_anomaly_detection")
        if not os.path.exists(dataset_dir):
            if download:
                self.download_dataset(root)
            else:
                raise ValueError(
                    "Dataset not found. Please set download=True to download the dataset."
                )

        if train:
            self.data = glob(
                os.path.join(dataset_dir, category, "train", "good", "*.png")
            )

        else:
            image_files = glob(
                os.path.join(dataset_dir, category, "test", "*", "*.png")
            )
            normal_image_files = glob(
                os.path.join(dataset_dir, category, "test", "good", "*.png")
            )
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.data = image_files

        self.data.sort(key=lambda y: y.lower())
        self.train = train

    def __getitem__(self, index):
        image_file = self.data[index]
        image = Image.open(image_file).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1

        return image, target

    def __len__(self):
        return len(self.data)

    def download_dataset(self, root):
        url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
        dataset_dir = os.path.join(root, "mvtec_anomaly_detection")

        # Create directory for dataset
        os.makedirs(dataset_dir, exist_ok=True)

        # Download and extract dataset
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024

        desc = "\033[33mDownloading MVTEC...\033[0m"
        progress_bar = tqdm(
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            desc=desc,
            position=0,
            leave=True,
        )

        with open(os.path.join(root, "mvtec_anomaly_detection.tar.xz"), "wb") as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)

        progress_bar.close()

        tar_command = [
            "tar",
            "-xf",
            os.path.join(root, "mvtec_anomaly_detection.tar.xz"),
            "-C",
            dataset_dir,
        ]

        subprocess.run(tar_command)


def get_MVTEC_train(normal_class_labels, path, backbone):
    transform = transform_color

    all_trainsets = []
    all_trainsets_moco = []

    for normal_class_indx in list(set(normal_class_labels)):
        normal_class = mvtec_labels[normal_class_indx]
        trainset = MVTecDataset(
            path, normal_class, transform, train=True, download=True
        )

        trainset_moco = MVTecDataset(
            path, normal_class, dino_transform, train=True, download=True
        )

        all_trainsets.append(trainset)
        all_trainsets_moco.append(trainset_moco)

    trainset = ConcatDataset(all_trainsets)
    trainset_moco = ConcatDataset(all_trainsets_moco)

    return trainset, trainset_moco


def get_MVTEC_test(normal_class_labels, path, backbone):
    transform = transform_color
    all_testsets = []

    for normal_class_indx in list(set(normal_class_labels)):
        normal_class = mvtec_labels[normal_class_indx]
        testset = MVTecDataset(
            path, normal_class, transform, train=False, download=True
        )
        all_testsets.append(testset)

    testset = ConcatDataset(all_testsets)
    return testset


class Cub2011(VisionDataset):
    """`CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
           creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
           and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
           target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
           puts it in root directory. If dataset is already downloaded, it is not
           downloaded again.
    """

    base_folder = "CUB_200_2011/images"
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    file_id = "1hbzc_P1FuxMkcabkgn9ZKinBwW683j45"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        super(Cub2011, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.loader = default_loader
        self.train = train
        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

    def _load_metadata(self):
        images = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )
        image_class_labels = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"],
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )

        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")

        class_names = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "classes.txt"),
            sep=" ",
            names=["class_name"],
            usecols=[1],
        )
        self.class_names = class_names["class_name"].to_list()
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_file_from_google_drive(
            self.file_id, self.root, self.filename, self.tgz_md5
        )

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class CUBAnomaly(Cub2011):
    def __init__(self, root, normal_classes, subclasses, **kwargs):
        super().__init__(root, **kwargs)
        self.normal_classes = normal_classes

        if subclasses is None:
            subclasses = list(range(200))

        self.subclasses = subclasses

        if self.train and not set(normal_classes).issubset(set(subclasses)):
            raise ValueError("Normal classes must be a subset of subclasses")

        classes = self.normal_classes if self.train else self.subclasses
        self.subset_classes(classes)

    def subset_classes(self, classes):
        # Filter data
        subclasses = np.array(classes)
        subset_data = self.data[self.data["target"].isin(subclasses + 1)]
        self.data = subset_data

        # Update attributes
        self.class_names = [self.class_names[i] for i in subclasses.tolist()]

    @property
    def num_classes(self):
        return len(self.class_names)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        label = target

        if self.train:
            if target in self.normal_classes:
                label = 0
        else:
            if target in self.normal_classes:
                label = 0
            else:
                label = 1

        return img, label


def get_CUB_train(normal_class_labels, path, backbone):
    transform = transform_color

    trainset = CUBAnomaly(
        root=path,
        train=True,
        download=True,
        normal_classes=normal_class_labels,
        subclasses=list(range(20)),
        transform=transform,
    )
    trainset_moco = CUBAnomaly(
        root=path,
        train=True,
        download=True,
        normal_classes=normal_class_labels,
        subclasses=list(range(20)),
        transform=dino_transform,
    )

    return trainset, trainset_moco


def get_CUB_test(normal_class_labels, path, backbone):
    transform = transform_color

    testset = CUBAnomaly(
        root=path,
        train=False,
        download=True,
        normal_classes=normal_class_labels,
        subclasses=list(range(20)),
        transform=transform,
    )

    return testset