from torchvision import datasets, transforms
from torch.utils.data import DataLoader


DATASET_NUM_LABELS = {
    "cifar10": 10,
    "cifar100": 100,
}


def get_vision_transforms(image_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def load_vision_dataset(dataset_name: str, data_dir: str = "./data_cache", image_size: int = 224):
    train_transform, val_transform = get_vision_transforms(image_size)

    if dataset_name == "cifar10":
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        val_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)
    elif dataset_name == "cifar100":
        train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        val_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=val_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_set, val_set


def get_vision_dataloaders(train_set, val_set, batch_size: int = 32, num_workers: int = 2):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
