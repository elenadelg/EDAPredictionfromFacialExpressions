import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
' Data Preprocessing Pipeline                                                       '
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

class DataPreprocessor:

    '''
    Preprocess frames according to DeepPhys paper pipeline: 
    1. Preprocessing single frames 
        - Downsample frames to LxL (L=36 recommended)
        - Scale pixel values to [0,1]
        - Normalize to zero mean and unit standard deviation
    2. Motion Representation
        - Compute normalized frame difference using
             Dl(t) = (C(t+1) - C(t)) / (C(t+1) + C(t))
        - Clip outliers by 3 standard deviations
        - Scale to unit standard deviation

    '''
    
    def __init__(self, base_path: str, L: int = 36):
        self.base_path = Path(base_path)
        self.L = L
        self.frame_dict = {} 
        self.patient_data = {}  
        
    def load_csv_data(self) -> None:

        for csv_file in self.base_path.glob("Dataset_*.csv"):
            filename_parts = csv_file.stem.split('_')
            patient_id = filename_parts[1]
            task_num = filename_parts[2]

            df = pd.read_csv(csv_file)

            if patient_id not in self.patient_data:
                self.patient_data[patient_id] = {}

            self.patient_data[patient_id][task_num] = {
                'frames': df['Frames'].tolist(),
                'bvp': df['bvp'].values,
                'eda': df['eda'].values
            }

            for frame, bvp, eda in zip(df['Frames'], df['bvp'], df['eda']):
                self.frame_dict[frame] = {'bvp': bvp, 'eda': eda}

            print(f"Loaded data for patient {patient_id}, task {task_num} with {len(df)} frames.")

    def get_frame_path(self, frame_name: str) -> str:
        return str(self.base_path / frame_name)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a frame by downsampling, standardizing, and normalizing pixel values.
        """
        frame = frame.astype(np.float32) / 255.0 # Convert to float32 and normalize to [0,1]
        frame = cv2.resize(frame, (self.L, self.L), interpolation=cv2.INTER_CUBIC) # Downsample to LxL - 36x36
        frame = (frame - np.mean(frame)) / (np.std(frame) + 1e-7) # standardize
        return frame

    def compute_normalized_frame_difference(self, 
                                            frame1: np.ndarray, 
                                            frame2: np.ndarray) -> np.ndarray:
        """
        Compute the normalized frame difference, which represents motion.
        Formula: Dl(t) = (Cl(t + Δt) - Cl(t)) / (Cl(t + Δt) + Cl(t)).
        """
        diff = frame2 - frame1
        sum_frames = frame2 + frame1 + 1e-7  
        norm_diff = diff / sum_frames 
        std = np.std(norm_diff)
        norm_diff = np.clip(norm_diff, -3*std, 3*std) # Clip outliers (3 standard deviations)
        norm_diff = norm_diff / (np.std(norm_diff) + 1e-7) # scale 
        return norm_diff

    def process_sequence(self, 
                        frames_list: List[str], 
                        patient_id: str, 
                        task_num: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a sequence of frames and their corresponding physiological signals.
        """
        processed_frames = []
        motion_representations = []
        eda_signals = []
        
        for i in range(len(frames_list)-1): 
            
            # Work with consecutive frames
            frame1_path = self.get_frame_path(frames_list[i])
            frame2_path = self.get_frame_path(frames_list[i+1])
            
            frame1 = cv2.imread(frame1_path)
            frame2 = cv2.imread(frame2_path)
            
            if frame1 is None or frame2 is None:
                print(f"Warning: Could not load frames: {frame1_path} or {frame2_path}")
                continue
                
            proc_frame1 = self.preprocess_frame(frame1) #preprocess
            proc_frame2 = self.preprocess_frame(frame2)
            
            motion_rep = self.compute_normalized_frame_difference(proc_frame1, proc_frame2) # normalized difference
            
            processed_frames.append(proc_frame1)
            motion_representations.append(motion_rep)
            eda_signals.append(self.frame_dict[frames_list[i]]['eda'])

        return np.array(processed_frames), np.array(motion_representations), np.array(eda_signals), patient_id, task_num

    def process_all_data(self) -> Dict:
        """
        Process all patients and their tasks.
        """
        processed_data = {}

        for patient_id, patient_tasks in self.patient_data.items():
            processed_data[patient_id] = {}
            
            for task_num, task_data in patient_tasks.items():
                frames_list = task_data['frames']
                
                processed_frames, motion_reps, eda, patient_id, task_num = self.process_sequence(frames_list, patient_id, task_num)
                
                processed_data[patient_id][task_num] = {
                    'frames': processed_frames,
                    'motion': motion_reps,
                    'eda': eda
                }

            print(f"Processed all tasks for patient {patient_id}.")
        return processed_data
    
    def visualize_preprocessing(self, 
                              frame_name: str) -> None:
        """
        Visualize the preprocessing steps for a single frame
        """
        frame_path = self.get_frame_path(frame_name)
        original_frame = cv2.imread(frame_path)
        
        if original_frame is None:
            print(f"Error: Could not load frame {frame_path}")
            return
            
        processed_frame = self.preprocess_frame(original_frame)
        
        motion_rep = None
        frame_parts = frame_name.split('_')
        frame_number = int(frame_parts[-1].replace('.jpg', ''))
        next_frame_name = f"{frame_parts[0]}_{frame_parts[1]}_frame_{frame_number + 1}.jpg"
        next_frame_path = self.get_frame_path(next_frame_name)
        
        if os.path.exists(next_frame_path):
            next_frame = cv2.imread(next_frame_path)
            if next_frame is not None:
                proc_next_frame = self.preprocess_frame(next_frame)
                motion_rep = self.compute_normalized_frame_difference(processed_frame, proc_next_frame)
        
        if motion_rep is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Frame')
        axes[1].imshow(processed_frame, cmap='gray')
        axes[1].set_title(f'Preprocessed Frame ({self.L}x{self.L})')
        
        if motion_rep is not None:
            axes[2].imshow(motion_rep, cmap='RdBu')
            axes[2].set_title('Motion Representation')
        
        plt.tight_layout()
        plt.show()




''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
' Tensor Creation                                                                  '
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

class DeepPhysTensor(Dataset):
    """Dataset class for preparing and storing tensors"""
    
    def __init__(self, processed_data: dict, sequence_length: int = 4, save_path: str = None):

        self.sequence_length = sequence_length # The length of the sequence to create for each sample (default is 4).
        self.save_path = save_path
         # Prepare the dataset by organizing the data into samples (sequences of frames and corresponding motion/EDA)
        self.samples = self._prepare_samples(processed_data)
        
        if save_path:
            self.save_tensors()
    
    def _prepare_samples(self, processed_data):
        """
        Prepare sequences from the processed data for each patient and task.
        Organize the data into sequences of frames, motion, and EDA.
        """
        samples = []
        
        for patient_id in processed_data:
            for task_num in processed_data[patient_id]: # process sequences just within each task -> each task is specific 
                data = processed_data[patient_id][task_num]
                frames = data['frames']
                motion = data['motion']
                eda = data['eda']
                
                # Add patient_id and task_num when creating each sample
                for i in range(len(frames) - self.sequence_length):
                    frame_seq = frames[i:i+self.sequence_length]
                    motion_seq = motion[i:i+self.sequence_length]
                    eda_seq = eda[i:i+self.sequence_length]
                    
                    samples.append({
                        'frames': frame_seq,
                        'motion': motion_seq,
                        'eda': eda_seq[-1],  # Using last EDA value as target
                        'patient_id': patient_id,  # Add patient_id
                        'task_num': task_num  # Add task_num
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):

        sample = self.samples[idx]

        frames = torch.from_numpy(sample['frames']).float()
        motion = torch.from_numpy(sample['motion']).float()
        
        # Expand single channel to 3 channels
        frames = frames.permute(0, 3, 1, 2)  # [seq_len, 3, L, L]
        motion = motion.permute(0, 3, 1, 2)  # [seq_len, 3, L, L]
        
        combined = torch.cat([motion[-1:], frames[-1:]], dim=1)  # [1, 6, L, L]
        
        return combined.squeeze(0), torch.tensor(sample['eda'], dtype=torch.float32), sample['patient_id'], sample['task_num']
    
    def save_tensors(self):
        """
        Save tensors for each patient are saved in a separate folder with the patient ID.
        """
        if not self.save_path:
            raise ValueError("save_path not specified")
        
        os.makedirs(self.save_path, exist_ok=True)
        tensor_info = []
        
        for idx in range(len(self)):
            input_tensor, target, patient_id, task_num = self[idx]
            
            patient_folder = os.path.join(self.save_path, f'patient_{patient_id}')
            os.makedirs(patient_folder, exist_ok=True)
            
            tensor_file_path = os.path.join(patient_folder, f"tensor_{idx}.pt")
            torch.save({
                'input': input_tensor,
                'target': target
            }, tensor_file_path)
            
            info = {
                'index': idx,
                'patient_id': patient_id,
                'task_num': task_num
            }
            tensor_info.append(info)
        
        tensor_info_file = os.path.join(self.save_path, 'tensor_info.pt')
        torch.save(tensor_info, tensor_info_file)

        for patient_id in set([sample['patient_id'] for sample in self.samples]):
            patient_folder = os.path.join(self.save_path, f'patient_{patient_id}')
            print(f"All tensors for patient {patient_id} have been saved in {patient_folder}")

        print(f"Saved {len(self)} tensors to {self.save_path}")



def process_and_save_data(raw_data_path: str, save_path: str):
    """Process raw data: data preprocessing pipeline"""
    # Initialize and run preprocessor
    preprocessor = DataPreprocessor(raw_data_path)
    preprocessor.load_csv_data()
    processed_data = preprocessor.process_all_data()
    
    """Create tensors data from processed data"""
    # Create dataset and save tensors
    dataset = DeepPhysTensor(processed_data, save_path=save_path)
    return dataset


