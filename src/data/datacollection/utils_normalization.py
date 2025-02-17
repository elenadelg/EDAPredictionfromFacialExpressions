import os
import pandas as pd


for file_path in csv_files:
    df = pd.read_csv(file_path)
    
    required_cols = {'Frames', 'bvp', 'eda'}
    if not required_cols.issubset(df.columns):
        print(f"Skipping {file_path}: missing required columns.")
        continue
    
    print(f"Normalizing: {file_path}")
    # Normalize 'bvp' and 'eda' columns using Min-Max normalization
    for col in ['bvp', 'eda']:
        min_val = df[col].min() 
        max_val = df[col].max() 
        if max_val - min_val != 0:
            df[col] = (df[col] - min_val) / (max_val - min_val) 
        else:
            df[col] = 0.0  

    filename = os.path.basename(file_path)
    new_file_path = os.path.join(destination_folder_path, filename)

    df.to_csv(new_file_path, index=False)

    print(f"Processed and normalized: {new_file_path}")
