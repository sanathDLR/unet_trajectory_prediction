"""
Author: Marko Mizdrak

This script downloads and extracts the DLR data set used in the project

"""
import os
import zipfile
import shutil
import requests
from pathlib import Path
from tqdm import tqdm

# Constants
DOWNLOAD_URL = "https://zenodo.org/records/15025237/files/DLR-Urban-Traffic-dataset_v1-2-1.zip?download=1"
ZIP_FILENAME = "DLR-Urban-Traffic-dataset_v1-2-1.zip"
EXTRACT_DIR = "DLR-Urban-Traffic-dataset"

def download_zip(url: str, dest: str) -> None:
    print(f"[INFO] Starting download from: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(dest, "wb") as f, tqdm(
        desc="[INFO] Downloading",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"[INFO] Download complete: {dest}")

def unzip_file(zip_path: str, extract_to: str) -> None:
    print(f"[INFO] Extracting ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[INFO] Extraction complete to: {extract_to}")

def find_trajectories_dir(root_dir: Path) -> Path | None:
    for root, dirs, _ in os.walk(root_dir):
        for d in dirs:
            if d.lower() == "trajectories":
                return Path(root) / d
    return None

def list_csv_files(directory: Path) -> list[Path]:
    return [Path(root) / file
            for root, _, files in os.walk(directory)
            for file in files if file.endswith(".csv")]

def copy_csv_files(csv_files: list[Path], target_dir: Path) -> None:
    total_files = len(csv_files)
    total_bytes = sum(f.stat().st_size for f in csv_files)

    print(f"[INFO] Found {total_files} CSV files ({total_bytes / 1e6:.2f} MB). Starting copy...")

    with tqdm(total=total_bytes, unit='B', unit_scale=True, unit_divisor=1024, desc="[INFO] Copying") as bar:
        for src_file in csv_files:
            dst_file = target_dir / src_file.name
            with open(src_file, 'rb') as src, open(dst_file, 'wb') as dst:
                while True:
                    chunk = src.read(8192)
                    if not chunk:
                        break
                    dst.write(chunk)
                    bar.update(len(chunk))

    print(f"[INFO] Copy complete: {total_files} files copied.")

def main():
    current_dir = Path.cwd()
    zip_path = current_dir / ZIP_FILENAME
    extract_path = current_dir / EXTRACT_DIR

    if not zip_path.exists():
        download_zip(DOWNLOAD_URL, zip_path)
    else:
        print(f"[INFO] ZIP file already exists: {zip_path}")

    if not extract_path.exists():
        unzip_file(zip_path, extract_path)
    else:
        print(f"[INFO] Extract directory already exists: {extract_path}")

    print(f"[INFO] Locating 'trajectories' folder...")
    trajectories_dir = find_trajectories_dir(extract_path)

    if not trajectories_dir or not trajectories_dir.exists():
        print("[ERROR] 'trajectories' folder not found.")
        return

    csv_files = list_csv_files(trajectories_dir)
    if not csv_files:
        print("[WARNING] No CSV files found in trajectories folder.")
        return

    copy_csv_files(csv_files, current_dir)

if __name__ == "__main__":
    main()
