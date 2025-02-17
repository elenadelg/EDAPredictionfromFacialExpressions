import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class Augmentation:
   '''
   Applies data augmentation to both motion and appearance tensors, including .
   - random horizontal flipping, 
   - rotation between -10 and 10 degrees,
   - brightness adjustments between 0.8x and 1.2x,
   - Gaussian noise injection with 0.02 standard deviation.
   Each augmentation is applied independently with probability p.

   Args:
       p (float): Probability of applying each augmentation, between 0 and 1
   '''
   def __init__(self, p=0.7):
       self.p = p
       
   def __call__(self, tensor, label):
       motion = tensor[:3]
       appearance = tensor[3:]
       
       if random.random() < self.p:
           motion = torch.flip(motion, [-1])
           appearance = torch.flip(appearance, [-1])
             
       if random.random() < self.p:
           angle = random.uniform(-10, 10)
           motion = transforms.functional.rotate(motion, angle)
           appearance = transforms.functional.rotate(appearance, angle)
           
       if random.random() < self.p:
           brightness_factor = random.uniform(0.8, 1.2)
           appearance = appearance * brightness_factor
           appearance = torch.clamp(appearance, -1, 1)
           
       if random.random() < self.p:
           noise = torch.randn_like(appearance) * 0.02
           appearance = appearance + noise
           appearance = torch.clamp(appearance, -1, 1)

       augmented_tensor = torch.cat([motion, appearance], dim=0)
       return augmented_tensor, label

class DeepPhysDataset(Dataset):
   '''
   Dataset for loading and optionally augmenting tensor data from .pt files. Handles both motion and appearance.  
   Optional: task filtering and data augmentation.

   Args:
       file_paths (list): List of paths to .pt files containing tensor data and labels
       tasks (str): can be 'T1', 'T2', 'T3' or 'all'
       augment (bool): 2hether to apply data augmentation
       augment_prob (float): Probability of applying each augmentation if augment=True

   Each .pt file should contain:
       - 'input': Tensor of shape [6, H, W] (3 channels motion + 3 channels appearance)
       - 'target': Target label tensor
   '''
   def __init__(self, file_paths, tasks="all", augment=False, augment_prob=0.7):
       self.file_paths = file_paths
       self.augmentation = Augmentation(p=augment_prob) if augment else None
       
       if tasks != "all":
           self.file_paths = [f for f in self.file_paths if f"_{tasks}_" in f]

   def __len__(self):
       return len(self.file_paths)

   def __getitem__(self, idx):
       data = torch.load(self.file_paths[idx])
       tensor = data['input']
       label = data['target'].view(1)
       
       if self.augmentation is not None:
           tensor, label = self.augmentation(tensor, label)
           
       return tensor, label

   @staticmethod
   def get_file_list(folder):
       '''Recursively collects paths of all .pt files in a directory and its subdirectories.
       '''
       return [
           os.path.join(root, file)
           for root, _, files in os.walk(folder)
           for file in files if file.endswith('.pt')
       ]

def create_dataloaders(train_dir, val_dir, test_dir, batch_size, tasks="all", augment=False, augment_prob=0.7):
   '''Creates DataLoader objects for training, validation and testing datasets.
   '''
   train_files = DeepPhysDataset.get_file_list(train_dir)
   val_files = DeepPhysDataset.get_file_list(val_dir)
   test_files = DeepPhysDataset.get_file_list(test_dir)

   train_dataset = DeepPhysDataset(
       train_files, 
       tasks=tasks,
       augment=augment,
       augment_prob=augment_prob
   )
   val_dataset = DeepPhysDataset(val_files, tasks=tasks, augment=False)
   test_dataset = DeepPhysDataset(test_files, tasks=tasks, augment=False)

   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
   test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

   return train_loader, val_loader, test_loader