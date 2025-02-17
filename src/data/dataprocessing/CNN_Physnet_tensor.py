import os
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TensorCreation:
    """
    Preprocesses raw frame images and their corresponding EDA and BVP, and saves them as `.pt` files.
    """
    def __init__(self, resize=(128, 128)):
        """
        transform to tensor, normalize, and resize according to typical PhysNet or CNN specs
        """
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.Resize(resize),  
            transforms.ToTensor(),  
            transforms.Normalize((0.5,), (0.5,))  
        ])

    def create_tensors(self, source_dir, target_dir):
        """
        Preprocesses raw frame images and EDA/BVP data from CSV files and saves them as `.pt` files.
        """
        os.makedirs(target_dir, exist_ok=True) 

        for file in os.listdir(source_dir):
            if file.endswith('.csv'):
                csv_path = os.path.join(source_dir, file)
                eda_bvp_df = pd.read_csv(csv_path)  
                
                filename_parts = os.path.splitext(file)[0].split('_')
                if len(filename_parts) < 3:
                    print(f"Skipping {file} - Unable to parse subject/session identifiers.")
                    continue
                s_identifier, t_identifier = filename_parts[1], filename_parts[2]

                for frame_file in os.listdir(source_dir):
                    if frame_file.startswith(f"{s_identifier}_{t_identifier}") and frame_file.endswith('.jpg'):
                        frame_path = os.path.join(source_dir, frame_file)
                        if not os.path.exists(frame_path):
                            print(f"Frame {frame_file} not found. Skipping...")
                            continue

                        try:
                            img = Image.open(frame_path).convert('RGB')
                            img_tensor = self.transform(img)

                            row = eda_bvp_df.loc[eda_bvp_df['Frames'] == frame_file]
                            if row.empty:
                                print(f"No matching EDA/BVP entry for {frame_file}. Skipping...")
                                continue
                            eda_value = row['eda'].values[0]
                            bvp_value = row['bvp'].values[0]

                            subfolder = os.path.join(target_dir, s_identifier)
                            os.makedirs(subfolder, exist_ok=True)

                            # Save the processed data as a `.pt` file
                            tensor_save_path = os.path.join(subfolder, frame_file.replace('.jpg', '.pt'))
                            data_entry = {
                                "tensor": img_tensor,
                                "bvp": bvp_value,
                                "eda": eda_value
                            }
                            torch.save(data_entry, tensor_save_path)
                            print(f"Processed and saved: {tensor_save_path}")

                        except Exception as e:
                            print(f"Failed to process {frame_path}: {e}")


class TensorDataset(Dataset):
    """
    A dataset for loading fixed-length sequences of frames and their physiological signals 
    (EDA and BVP) from preprocessed .pt files.

    Inputs:
        file_paths (list of str): A list of paths to .pt files, each containing:
            - "tensor": A PyTorch tensor of shape (C, H, W).
            - "eda"   : Scalar value representing EDA.
            - "bvp"   : Scalar value representing BVP.

    Outputs (from __getitem__):
        tuple: (sequence_tensor, eda_target, bvp_target)
            - sequence_tensor (torch.Tensor): Shape (C, sequence_length, H, W).
            - eda_target (torch.Tensor)     : Scalar tensor of the last frame's EDA.
            - bvp_target (torch.Tensor)     : Scalar tensor of the last frame's BVP.
    """
    
    def __init__(self, file_paths, sequence_length=4):
        self.file_paths = file_paths
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.file_paths) - self.sequence_length + 1

    def __getitem__(self, idx):
        sequence_files = self.file_paths[idx : idx + self.sequence_length]
        
        sequence_frames = []
        eda_targets = []
        bvp_inputs = []

        for file in sequence_files:
            data = torch.load(file)
            sequence_frames.append(data['tensor'])  # (C, H, W)
            eda_targets.append(data['eda'])  
            bvp_inputs.append(data['bvp'])  

        # Stack frames along a new dimension: (C, sequence_length, H, W)
        sequence_tensor = torch.stack(sequence_frames, dim=1)

        return (
            sequence_tensor, 
            torch.tensor(eda_targets[-1], dtype=torch.float32), 
            torch.tensor(bvp_inputs[-1], dtype=torch.float32)
        )


def process_and_save_data(raw_data_path: str, save_path: str, sequence_length=4) -> TensorDataset:
    """
    Gathers the .pt files and creates a TensorDataset of the given sequence length.
    """
    preprocessor = TensorCreation(resize=(128, 128))
    preprocessor.create_tensors(source_dir=raw_data_path, target_dir=save_path)

    pattern = os.path.join(save_path, '**', '*.pt')
    all_pt_files = sorted(glob.glob(pattern, recursive=True))

    if not all_pt_files:
        print(f"No .pt files found in {save_path}. Please check your data.")
        return None

    dataset = TensorDataset(file_paths=all_pt_files, sequence_length=sequence_length)

    return dataset
