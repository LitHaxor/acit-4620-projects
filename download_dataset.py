"""
Download and unpack the PM25Vision dataset from Hugging Face into the format
expected by this repository:

datasets/
├── train/
│   ├── images/         # <filename>.jpg
│   └── metadata.csv
└── test/
    ├── images/         # <filename>.jpg
    └── metadata.csv
"""

import sys
from pathlib import Path

from tqdm import tqdm

try:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Missing required libraries. Please install them with:")
    print("    pip install huggingface_hub pyarrow")
    sys.exit(1)

REPO_ID = "DeadCardassian/PM25Vision"
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

SPLITS = ["test", "train"]

def process_split(split_name: str):
    print("\n==========================================")
    print(f"Downloading {split_name}.parquet from Hugging Face...")
    print("==========================================")
    
    parquet_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{split_name}.parquet",
        repo_type="dataset"
    )
    print(f"Downloaded to cache: {parquet_path}")

    split_dir = DATASETS_DIR / split_name
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {split_name} parquet table...")
    table = pq.read_table(parquet_path)
    
    # 1. Save metadata (all columns except the raw 'image' bytes)
    metadata_df = table.drop(["image"]).to_pandas()
    metadata_path = split_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Saved metadata: {metadata_path} ({len(metadata_df):,} rows)")

    # 2. Extract and write images to datasets/<split>/images/<filename>
    print(f"Extracting {len(metadata_df):,} images to {images_dir}...")
    filenames = table["filename"]
    images = table["image"]

    for i in tqdm(range(len(table)), desc=f"Writing {split_name} images"):
        img_bytes = images[i].as_py()
        fn = filenames[i].as_py()
        (images_dir / fn).write_bytes(img_bytes)

    print(f"Completed {split_name}: {len(list(images_dir.glob('*.jpg'))):,} images on disk.")

def main():
    print("PM25Vision Dataset Downloader")
    print(f"Destination: {DATASETS_DIR}")
    
    for split in SPLITS:
        process_split(split)

    print("\n All splits successfully downloaded and extracted!")

if __name__ == "__main__":
    main()
