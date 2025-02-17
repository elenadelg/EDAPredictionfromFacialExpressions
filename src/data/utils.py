import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
from scipy.signal import find_peaks
import random

# video and frames utils
def open_video(video_path):
    """
    Opens a video file and returns the VideoCapture object.

    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    return cap

def get_video_info(folder_path, frames_per_second=5):
    """
    Retrieves and prints video information for all video files in a folder: frame count, FPS, and duration.

    Parameters:
        folder_path (str): Path to the folder containing video files.
        frames_per_second (int): Number of frames per second used for extraction.
    """
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4'))]

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        cap = open_video(video_path)
        if not cap:
            continue

        # Get total number of frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get frames per second (FPS)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate video duration
        if fps > 0:
            duration = frame_count / fps
        else:
            print("Error: FPS value is zero, cannot calculate duration.")
            duration = None

        # Calculate total number of extracted frames
        if fps > 0:
            extracted_frame_count = int(duration * frames_per_second)
        else:
            extracted_frame_count = None

        cap.release()

        print(f"Video: {video_file}")
        print(f"Total number of frames: {frame_count}")
        print(f"Frames per second (FPS): {fps}")
        print(f"Duration of the video: {duration} seconds")
        print(f"Total number of extracted frames (at {frames_per_second} fps): {extracted_frame_count}\n")


def save_display_frames(folder_path, target_size=(240, 240)):
    """
    Saves 5 frames from the video file in the given folder, resizes them, converts them to grayscale,
    and saves them in a dedicated folder named 'frames_output'. Also displays the saved frames.

    Parameters:
    - folder_path (str): Path to the folder containing the video file.
    - target_size (tuple): Target size for each saved frame (default is 240x240).
    """

    output_folder = os.path.join(folder_path, 'frames_output')
    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4'))]

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file: {video_file}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < 5:
            print(f"Video {video_file} has less than 5 frames. Skipping.")
            cap.release()
            continue
        frame_interval = total_frames // 5
        frame_count = 0
        saved_frames = 0
        frames_to_display = []

        while saved_frames < 5:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()

            if not ret:
                break

            resized_frame = cv2.resize(frame, target_size)
            frame_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            
            frame_filename = f"{os.path.splitext(video_file)[0]}_frame_{saved_frames + 1}.png"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame_gray)
            print(f"Saved frame {saved_frames + 1} from {video_file} to {frame_path}")
            frames_to_display.append(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            saved_frames += 1
            frame_count += frame_interval

        cap.release()

        if frames_to_display:
            fig, axes = plt.subplots(1, 5, figsize=(20, 5))
            fig.suptitle(f"Frames from {video_file}")
            for i, frame in enumerate(frames_to_display):
                axes[i].imshow(frame)
                axes[i].axis('off')
                axes[i].set_title(f"Frame {i + 1}")
            plt.show()



# csv utils 
def read_csv_data(csv_file_path):
    """
    Reads BVP data from a CSV file.
    
    """
    import pandas as pd
    csv_data = pd.read_csv(csv_file_path, header=None)  
    # Return the first column as a list
    return csv_data.iloc[:, 0].tolist()  


def plot_csv_timeseries(folder_path, signal):
    """
    Plots time series and density plots for each CSV file in the given folder.

    """
    if signal == "bvp":
        print(f"BVP signals are sampled at 64 Hz i.e. 64 signals per second")
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f.startswith('bvp')]
        sampling_rate = 64
    elif signal == "eda":
        print(f"EDA signals are sampled at 4 Hz i.e. 4 signals per second")
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f.startswith('eda')]
        sampling_rate = 4
    else:
        print("Unsupported signal type. Please use 'bvp' or 'eda'.")
        return

    for file in files:
        file_path = os.path.join(folder_path, file)
        data = read_csv_data(file_path)
        num_signals = len(data)
        duration = num_signals / sampling_rate
        
        print(f"File path: {file_path}")
        print(f"Total number of signals: {num_signals}")
        print(f"Number of signals per second: {sampling_rate}; Video Duration: {duration:.2f} seconds")

        # Create subplots for time series and density plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        fig.suptitle(f"Plots for {file}", fontsize=16, fontweight='bold')

        # Time Series Plot
        time = [i / sampling_rate for i in range(num_signals)]
        sns.lineplot(ax=axes[0], x=time, y=data, color='darkred' if signal == 'bvp' else 'blue', linewidth=1)
        axes[0].set_xlabel('Time (seconds)', fontsize=12)
        axes[0].set_ylabel(f'{signal.upper()} Value', fontsize=12)
        axes[0].set_title(f'{signal.upper()} Signals over Time', fontsize=14)
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # Density Plot
        sns.kdeplot(ax=axes[1], data=data, color='darkred' if signal == 'bvp' else 'blue', fill=True, linewidth=0.5, alpha=0.4)
        axes[1].set_xlabel(f'{signal.upper()} Value', fontsize=12)
        axes[1].set_title(f'Density of {signal.upper()} Values', fontsize=14)
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def peaks_and_frames(folder_path, signal, threshold=0.5):

    """
    Extracts frames corresponding to significant peaks in physiological signals (EDA or BVP). 
    Peaks are identified from CSV data files, and the frames at the times of these peaks are displayed.

    Parameters:
    - folder_path (str): Path to the folder containing video and CSV files.
    - signal (str): Type of signal to analyze ('eda' or 'bvp').
    - threshold (float): Threshold multiplier for detecting significant peaks in the signal.

    """
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4'))]

    if signal == "eda":
        print(f"EDA peaks and their corresponding frames:")
        sampling_rate = 4
        line_color = 'blue'
    elif signal == "bvp":
        print(f"BVP peaks and their corresponding frames:")
        sampling_rate = 64
        line_color = 'darkred'
    else:
        print("Unsupported signal type. Please use 'eda' or 'bvp'.")
        return

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file: {video_path}")
            continue
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        identifier = '_'.join(video_file.split('_')[1:]).rsplit('.', 1)[0]
        file_name = f"{signal}_{identifier}.csv"
        
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            print(f"Corresponding CSV file not found for video: {video_file}")
            continue

        data = read_csv_data(file_path)
        signals = np.array(data)

        peaks, _ = find_peaks(signals, height=np.mean(signals) + np.std(signals) * threshold)

        plt.figure(figsize=(8, 3))
        plt.plot(signals, label=f'{signal.upper()} Signal', color=line_color)
        plt.plot(peaks, signals[peaks], 'x', color='black', label='Peaks')
        plt.title(f'{signal.upper()} Signal with Peaks for {video_file}')
        plt.xlabel('Time (samples)')
        plt.ylabel(f'{signal.upper()} Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        times = [peak / sampling_rate for peak in peaks] 
        plt.figure(figsize=(20, 10))
        num_frames = len(times)
        for i, time in enumerate(times):
            frame_number = int(time * video_fps)  # Calculate the corresponding frame number
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                plt.subplot((num_frames // 5) + 1, 5, i + 1)
                plt.imshow(frame_rgb)
                plt.axis('off')
                plt.title(f"Time: {time:.2f} seconds")
            else:
                print(f"Failed to extract frame at {time:.2f} seconds from {video_file}.")

        plt.tight_layout()
        plt.show()

        cap.release()



# frames extraction utils
def extract_frames(folder_path, frames_per_second=5, target_size=(240, 240)):
    """
    Extracts frames from all videos in folder_path at a specified rate, resizes them, and converts them to grayscale.
    The frames are not saved to disk but are returned as a list.

    Parameters:
    - folder_path (str): Path to the folder containing video files.
    - frames_per_second (int): Number of frames to extract per second (default is 1).
    - target_size (tuple): Target size for each saved frame (default is 240x240).

    Returns:
    - dict: A dictionary with video file names as keys and lists of frames as values.
    """
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4'))]
    extracted_frames = {}

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        cap = open_video(video_path)
        if not cap:
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / frames_per_second)
        
        frame_count = 0
        frames_list = []

        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                resized_frame = cv2.resize(frame, target_size)
                frame_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                frames_list.append(frame_gray)
                print(f"Extracted frame from {video_file} at count {frame_count}")
            frame_count += 1

        extracted_frames[video_file] = frames_list
        cap.release()

    return extracted_frames

def display_extracted_frames(extracted_frames, frame_limit=5):
    """
    Displays a specified number of frames from the extracted frames dictionary using Matplotlib.
    
    Parameters:
        extracted_frames (dict): Dictionary with video file names as keys and lists of frames as values.
        frame_limit (int): Number of frames to display from each video.
    """
    for video_file, frames in extracted_frames.items():
        print(f"Displaying frames for video: {video_file}")
        for i in range(min(frame_limit, len(frames))):
            frame = frames[i]
            plt.imshow(frame, cmap='gray')  # Since frames are in grayscale
            plt.axis('off')
            plt.show()



# visualization utils

def extract_frames_with_bvp(folder_path, frames_per_second=5, target_size=(240, 240)):
    """
    Extracts frames from all videos in a folder at a specified rate, resizes them, converts them to grayscale, 
    and synchronizes them with BVP data.
    
    Parameters:
    - folder_path (str): Path to the folder containing video files and corresponding BVP CSV files.
    - frames_per_second (int): Number of frames to extract per second (default is 1).
    - target_size (tuple): Target size for each saved frame (default is 240x240).

    Returns:
    - dict: A dictionary with video file names as keys and tuples of (frames, bvp_signals) as values.
    """
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4'))]
    extracted_data = {}

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        # Use the full identifier (e.g., 's2_T1') to match the corresponding BVP file
        identifier = '_'.join(video_file.split('_')[1:]).rsplit('.', 1)[0]
        bvp_file_name = f"bvp_{identifier}.csv"
        bvp_file_path = os.path.join(folder_path, bvp_file_name)
        print(f"video path: {video_file}; file path: {bvp_file_path}")

        if not os.path.exists(bvp_file_path):
            identifier = '_'.join(video_file.split('_')[1:3])  # Match the 's2_T1' part precisely
            bvp_file_name = f"bvp_{identifier}.csv"
            bvp_file_path = os.path.join(folder_path, bvp_file_name)

        if not os.path.exists(bvp_file_path):
            print(f"BVP file not found for video {video_file}. Expected: {bvp_file_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file: {video_file}")
            continue

        bvp_signal = read_csv_data(bvp_file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
    
        frame_interval = int(fps / frames_per_second)
        frame_count = 0
        frames_list = []
        bvp_list = []

        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                resized_frame = cv2.resize(frame, target_size)
                frame_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                frames_list.append(frame_gray)

                time_in_seconds = frame_count / fps
                bvp_index = min(int(time_in_seconds * len(bvp_signal) / (cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)), len(bvp_signal) - 1)
                bvp_list.append(bvp_signal[bvp_index])
                print(f"Extracted frame from {video_file} at count {frame_count} with BVP value {bvp_signal[bvp_index]}")
            frame_count += 1

        extracted_data[video_file] = (frames_list, bvp_list)
        cap.release()

    return extracted_data


def display_random_extracted_frames_with_bvp(extracted_data, frame_limit=5):
    """
    Displays a specified number of random extracted frames and corresponding BVP values.
    
    Parameters:
        extracted_data (dict): Dictionary with video file names as keys and tuples of (frames, bvp_signals) as values.
        frame_limit (int): Number of frames to display for each video.
    """
    for video_file, (frames, bvps) in extracted_data.items():
        print(f"Displaying random frames for video: {video_file}")
        indices = random.sample(range(len(frames)), min(frame_limit, len(frames)))
        for i in indices:
            frame = frames[i]
            bvp_value = bvps[i]
            plt.imshow(frame, cmap='gray')
            plt.title(f"BVP Value: {bvp_value}")
            plt.axis('off')
            plt.show()


def plot_eda_bvp_from_pt_files(directory_path):
    """
    This function goes through all .pt files, extracts values for 'eda' and 'bvp' keys,
    and plots their range.
    """
    eda_values = []
    bvp_values = []

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".pt") and "s13_T1" in file_name:
            file_path = os.path.join(directory_path, file_name)
            
            data = torch.load(file_path)
        
            if 'eda' in data and 'bvp' in data:
                eda_value = data['eda']
                bvp_value = data['bvp']
                
                if isinstance(eda_value, (list, torch.Tensor)):
                    eda_values.extend(eda_value)
                else:
                    eda_values.append(eda_value)
                
                if isinstance(bvp_value, (list, torch.Tensor)):
                    bvp_values.extend(bvp_value)
                else:
                    bvp_values.append(bvp_value)
    
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(eda_values, label='EDA', color='blue')
    plt.title('EDA Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(bvp_values, label='BVP', color='red')
    plt.title('BVP Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.show()


# patients info utils 

def plot_patient_info(csv_path):
    """
    Generate plots based on the InfoPatients.csv file.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Plot gender distribution
    gender_counts = df['Gender'].value_counts()
    plt.figure(figsize=(8, 6))
    ax = gender_counts.plot(kind='bar', color='skyblue')
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, count in enumerate(gender_counts):
        ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10, color='black')
    plt.show()
    
    # Plot scenario distribution (Experience column)
    scenario_counts = df['Experience'].value_counts()
    plt.figure(figsize=(8, 6))
    ax = scenario_counts.plot(kind='bar', color='orange')
    plt.title('Scenario Distribution')
    plt.xlabel('Scenario')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, count in enumerate(scenario_counts):
        ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10, color='black')
    plt.show()
    
    # Plot date distribution
    date_counts = df['Date'].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    date_counts.plot(kind='bar', color='lightgreen')
    plt.title('Date Distribution')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    # Plot time distribution
    time_counts = df['Time'].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    time_counts.plot(kind='bar', color='lightcoral')
    plt.title('Time Distribution')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def print_sorted_csv_as_table(csv_path):
    """
    Load and print the CSV file as a sorted, nicely formatted table.
    Sorting is done by 'Experience' and 'Gender' columns.
    """

    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values(by=['Experience', 'Gender'])

    print("\nSorted InfoPatients.csv Table:\n")
    print(df_sorted.to_string(index=False))


def display_random_image_per_patient(base_path):
    """
    Display one random image for each patient in the specified folder.
    """
    patient_images = {}
    for file_name in os.listdir(base_path):
        if file_name.endswith('.jpg'):
            patient_id = file_name.split('_')[0]  # e.g., 's1' from 's1_T3_frame_0015.jpg'
            
            if patient_id not in patient_images:
                patient_images[patient_id] = []
            patient_images[patient_id].append(os.path.join(base_path, file_name))

    for patient_id, images in patient_images.items():
        random_image_path = random.choice(images)  # Select a random image
        image = Image.open(random_image_path)

        plt.figure()
        plt.imshow(image)
        plt.title(f"Patient: {patient_id}")
        plt.axis('off')
        plt.show()