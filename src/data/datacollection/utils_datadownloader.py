import requests
from tqdm import tqdm
import tempfile
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_and_extract_dataset(email, password, dataset_url, output_dir):

    """
    This function logs in to the IEEE Dataport with the provided credentials, downloads the dataset from the given URL (IEEE Dataport),
    and extracts the content to the specified output directory.
    """

    # Login 
    login_url = "https://ieee-dataport.org/user/login"
    session = requests.Session()
    payload = {
        "name": email,
        "pass": password,
        "form_id": "user_login_form"
    }
    login_response = session.post(login_url, data=payload)
    if login_response.ok:
        dataset_name = dataset_url.split('/')[5].split('?')[0]
        print(f"Logged in successfully for patient: {dataset_name}")
    else:
        print("Failed to log in. Please check your credentials.")
        return
    
    # Extract zip files
    response = session.get(dataset_url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with tqdm(
            desc=f"Downloading zip file: {dataset_name}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                # A temporary file is used to store the downloaded zip file, ensuring it does not overwrite any existing files and is managed safely."
                for chunk in response.iter_content(chunk_size=16384):  # Increased chunk size
                    temp_file.write(chunk)
                    bar.update(len(chunk))
                temp_file_path = temp_file.name
        print("Download complete! Extracting files from the zip...")

        # Unzip the files to get content 
        with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
            zip_ref.extractall(Path(output_dir))
        print(f"Files extracted to {output_dir}")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")

def download_and_extract_datasets(email, password, dataset_urls, output_dir):
    # Use ThreadPoolExecutor to download datasets in parallel
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(download_and_extract_dataset, email, password, dataset_url, output_dir)
            for dataset_url in dataset_urls
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

