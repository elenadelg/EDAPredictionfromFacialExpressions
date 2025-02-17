import os
import glob
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF_rotate

class CNNDataset(Dataset):
    """
    A dataset that handles the loading of sequential data from .pt files and organizes them into
    windows of fixed size with a specified stride for sequential learning tasks.
    """
    def __init__(self, root_dir, window_size=16, stride=16, transform=None, apply_smoothing=False, smoothing_kernel_size=3):
        """
        Args:
            root_dir (str): Root directory containing .pt files.
            window_size (int): Number of frames per window.
            stride (int): Step size between consecutive windows.
            transform (callable, optional): Transformation applied to each frame.
            apply_smoothing (bool): Whether to apply temporal smoothing.
            smoothing_kernel_size (int): Kernel size for temporal smoothing.
        """
        super().__init__()
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.apply_smoothing = apply_smoothing
        self.smoothing_kernel_size = smoothing_kernel_size

        # Gather all .pt files and sort them by frame number
        pattern = os.path.join(root_dir, '**', '*.pt')
        all_files = sorted(glob.glob(pattern, recursive=True), key=self.parse_frame_number)
        
        if not all_files:
            print(f"[Warning] No .pt files found in {root_dir}!")

        self.filepaths = all_files

        # Build the indices for windowing
        self.indices = []
        N = len(self.filepaths)
        i = 0
        while i + window_size <= N:
            self.indices.append(i)
            i += stride

    @staticmethod
    def parse_frame_number(filename):
        basename = os.path.basename(filename)
        frame_str = basename.split('_')[-1].replace('.pt', '')
        return int(frame_str)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.window_size

        window_frames, window_bvp, window_eda = [], [], []

        for fpath in self.filepaths[start_idx:end_idx]:
            data = torch.load(fpath)
            frame, bvp, eda = data['tensor'], data['bvp'], data['eda']

            # Convert bvp and eda to tensors if necessary
            bvp = torch.tensor(bvp, dtype=torch.float32) if not isinstance(bvp, torch.Tensor) else bvp
            eda = torch.tensor(eda, dtype=torch.float32) if not isinstance(eda, torch.Tensor) else eda

            # Apply frame transformations if specified
            if self.transform:
                frame = self.transform(frame)

            window_frames.append(frame)
            window_bvp.append(bvp)
            window_eda.append(eda)

        # Stack into tensors
        window_frames = torch.stack(window_frames, dim=0)
        window_bvp = torch.stack(window_bvp, dim=0).squeeze()
        window_eda = torch.stack(window_eda, dim=0).squeeze()

        # Apply temporal smoothing if enabled
        if self.apply_smoothing:
            window_bvp = self.temporal_smoothing(window_bvp, self.smoothing_kernel_size)
            window_eda = self.temporal_smoothing(window_eda, self.smoothing_kernel_size)

        return window_frames, window_bvp, window_eda

    @staticmethod
    def temporal_smoothing(signal, kernel_size):
        signal = signal.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, T)
        kernel = torch.ones(1, 1, kernel_size) / kernel_size
        smoothed_signal = F.conv1d(signal, kernel, padding=(kernel_size // 2))
        return smoothed_signal.squeeze(0).squeeze(0)



class AugmentedDataset(Dataset):
    """
    A dataset wrapper that applies augmentations to the data returned by a base dataset. 
    """
    def __init__(self, base_dataset, isTrain=False, p=0.5):
        self.base_dataset = base_dataset
        self.isTrain = isTrain
        self.p = p

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        window_frames, window_bvp, window_eda = self.base_dataset[idx]

        # Apply augmentations during training
        if self.isTrain and random.random() < self.p:
            window_frames, window_bvp, window_eda = self.apply_augmentations(window_frames, window_bvp, window_eda)

        return window_frames, window_bvp, window_eda

    def apply_augmentations(self, frames, bvp, eda):
        if random.random() < self.p:
            frames = torch.flip(frames, dims=[-1])

        if random.random() < self.p:
            angle = random.uniform(-10, 10)
            frames = self.rotate_frames(frames, angle)

        if random.random() < self.p:
            brightness_factor = random.uniform(0.8, 1.2)
            frames = frames * brightness_factor
            frames = torch.clamp(frames, -1, 1)

        if random.random() < self.p:
            noise = torch.randn_like(frames) * 0.02
            frames = frames + noise
            frames = torch.clamp(frames, -1, 1)

        return frames, bvp, eda

    @staticmethod
    def rotate_frames(frames, angle):
        return torch.stack([TF_rotate(frame, angle) for frame in frames], dim=0)



def create_dataloaders(train_dir, val_dir, test_dir, batch_size, window_size=16, stride=16, 
                       apply_smoothing=False, smoothing_kernel_size=3, augment=False, augment_prob=0.7):
    """
    Creates DataLoader objects for training, validation, and testing datasets
    """

    train_window_dataset = CNNDataset(
        root_dir=train_dir,
        window_size=window_size,
        stride=stride,
        apply_smoothing=apply_smoothing,
        smoothing_kernel_size=smoothing_kernel_size
    )

    val_window_dataset = CNNDataset(
        root_dir=val_dir,
        window_size=window_size,
        stride=stride,
        apply_smoothing=apply_smoothing,
        smoothing_kernel_size=smoothing_kernel_size
    )

    test_window_dataset = CNNDataset(
        root_dir=test_dir,
        window_size=window_size,
        stride=stride,
        apply_smoothing=apply_smoothing,
        smoothing_kernel_size=smoothing_kernel_size
    )

    # apply augmentation to training  if required
    if augment:
        train_dataset = AugmentedDataset(
            base_dataset=train_window_dataset,
            isTrain=True,
            p=augment_prob
        )
    else:
        train_dataset = train_window_dataset
    val_dataset = val_window_dataset
    test_dataset = test_window_dataset

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
