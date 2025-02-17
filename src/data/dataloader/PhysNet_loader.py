import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class Augmentation:
    """
    Applies random augmentations to the input tensor with a probability of p.
    Augmentations include:
      - Random horizontal flip
      - Random rotation
      - Random brightness adjustment
      - Random noise injection
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor, eda, bvp):
        # Random horizontal flip
        if random.random() < self.p:
            tensor = torch.flip(tensor, [-1])  

        # Random rotation (-10 to 10 degrees)
        if random.random() < self.p:
            angle = random.uniform(-10, 10)
            tensor = F.rotate(tensor, angle)

        # Random brightness adjustment
        if random.random() < self.p:
            brightness_factor = random.uniform(0.8, 1.2)
            tensor = tensor * brightness_factor
            tensor = torch.clamp(tensor, -1, 1)

        # Random noise injection
        if random.random() < self.p:
            noise = torch.randn_like(tensor) * 0.02
            tensor = tensor + noise
            tensor = torch.clamp(tensor, -1, 1)

        return tensor, eda, bvp


class PhysNetDataset(Dataset):
    """
    This dataset loads single-frame .pt files where each file has:
      - 'tensor': shape (3, H, W)
      - 'eda': float
      - 'bvp': float

    """
    def __init__(self, file_paths, tasks="all", augment=False, augment_prob=0.7):
        self.file_paths = sorted(file_paths)
        self.augmentation = Augmentation(p=augment_prob) if augment else None
    
        if tasks != "all":
            self.file_paths = [f for f in self.file_paths if f"_{tasks}_" in f]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
       
        file_path = self.file_paths[idx]
        data = torch.load(file_path)   # => { 'tensor': (3, H, W), 'bvp': float, 'eda': float }

        image_2d = data['tensor']     # shape (3, H, W)
        # Insert a dummy time dimension => (3, 1, H, W)
        image_3d = image_2d.unsqueeze(1)

        eda = torch.tensor(data['eda'], dtype=torch.float32)
        bvp = torch.tensor(data['bvp'], dtype=torch.float32)

        if self.augmentation:
            image_3d, eda, bvp = self.augmentation(image_3d, eda, bvp)

        return image_3d, eda, bvp

    @staticmethod
    def get_file_list(folder):
       '''
       Recursively collects paths of all .pt files in a directory and its subdirectories.
       '''
       return [
           os.path.join(root, file)
           for root, _, files in os.walk(folder)
           for file in files if file.endswith('.pt')
       ]


def create_dataloaders(train_dir, val_dir, test_dir, batch_size, tasks="all", augment=False, augment_prob=0.7):
   '''
   Creates DataLoader objects for training, validation and testing datasets.
   '''
   train_files = PhysNetDataset.get_file_list(train_dir)
   val_files = PhysNetDataset.get_file_list(val_dir)
   test_files = PhysNetDataset.get_file_list(test_dir)

   train_dataset = PhysNetDataset(
       train_files, 
       tasks=tasks,
       augment=augment,
       augment_prob=augment_prob
   )
   val_dataset = PhysNetDataset(val_files, tasks=tasks, augment=False)
   test_dataset = PhysNetDataset(test_files, tasks=tasks, augment=False)

   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
   test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

   return train_loader, val_loader, test_loader
