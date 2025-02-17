import os
import logging
import pandas as pd
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def process_folder(folder, dataset_dir, output_dir):
    """
    This function processes a single folder, iterating over all video files in the folder,
    and extracts frames from each video based on the corresponding EDA and BVP signals.
    Then it creates a dataframe with corresponding frame name, BVP and EDA signal.
    """
    folder_path = os.path.join(dataset_dir, folder)
    logging.info(f"Processing folder: {folder}")
    if not os.path.isdir(folder_path):
        return

    video_files = [f for f in os.listdir(folder_path) if f.endswith('.avi')]
    if not video_files:
        logging.warning(f"No video files found in folder: {folder}. Skipping...")
        return

    # Iterate over all video files in the folder
    for selected_video in video_files:
        video_path = os.path.join(folder_path, selected_video)
        logging.info(f"Processing video: {selected_video}")

        # Determine the corresponding EDA and BVP files
        base_name = os.path.splitext(selected_video)[0]
        T_part = base_name.split('_')[-1]  # Extract T1, T2, T3
        eda_file = f'eda_{folder}_{T_part}.csv'
        bvp_file = f'bvp_{folder}_{T_part}.csv'
        eda_path = os.path.join(folder_path, eda_file)
        bvp_path = os.path.join(folder_path, bvp_file)
        logging.info(f"Looking for corresponding EDA file: {eda_file}")
        logging.info(f"Looking for corresponding BVP file: {bvp_file}")

        if not os.path.exists(eda_path):
            logging.warning(f"EDA file not found for {selected_video}. Skipping...")
            continue

        if not os.path.exists(bvp_path):
            logging.warning(f"BVP file not found for {selected_video}. Skipping...")
            continue

        # Load EDA and BVP signals: downsample BVP signals to match EDA sampling rate
        try:
            eda_data = pd.read_csv(eda_path, header=None)
            bvp_data = pd.read_csv(bvp_path, header=None)
        except Exception as e:
            logging.error(f"Failed to read EDA/BVP files. Error: {e}")
            continue
        eda_sampling_rate = 4  # 4 Hz
        bvp_sampling_rate = 64  # 64 Hz
        averaging_factor = int(bvp_sampling_rate / eda_sampling_rate) 
        bvp_downsampled = bvp_data.iloc[:, 0].groupby(bvp_data.index // averaging_factor).mean().reset_index(drop=True)
        logging.info(f"BVP data downsampled. Averaging factor: {averaging_factor}")

        # Open video and extract frames
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) 
        video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps 
        frame_timestamps = np.linspace(0, video_length, num=len(eda_data)) # to synchronize video frames with EDA data.
        frame_timestamps[-1] = min(frame_timestamps[-1], video_length - 1.0 / video_fps)
        all_frames_data = []  # List to store frame filenames and corresponding signals
        for i, timestamp in enumerate(frame_timestamps, start=1):
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Error reading frame at timestamp {timestamp:.2f} seconds. Skipping...")
                break
            frame_filename = f"{folder}_{T_part}_frame_{i:04d}.jpg"
            output_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(output_path, frame)
            if i <= len(eda_data) and i <= len(bvp_downsampled):
                all_frames_data.append((frame_filename, eda_data.iloc[i - 1, 0], bvp_downsampled.iloc[i - 1]))

        cap.release()

        # Create dataframe
        frames_df = pd.DataFrame(all_frames_data, columns=['Frames', 'eda', 'bvp'])
        output_csv_path = os.path.join(output_dir, f"Dataset_{folder}_{T_part}.csv")
        frames_df.to_csv(output_csv_path, index=False)
        logging.info(f"Dataset saved to: {output_csv_path}")


def process_folders_parallel(dataset_dir, output_dir):
    """
    This function processes all folders in parallel by using a thread pool.
    """
    folders = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]
    with ThreadPoolExecutor() as executor:
        executor.map(lambda folder: process_folder(folder, dataset_dir, output_dir), folders)
