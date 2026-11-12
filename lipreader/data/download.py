"""Download the GRID speaker s1 subset (videos + alignments, ~400 MB) into DATA_ROOT.

The archive is the LipNet-tutorial copy of GRID: data/s1/*.mpg and data/alignments/s1/*.align.
The full corpus (34 speakers, CC BY 4.0) is at https://zenodo.org/records/3625687.
"""
import os
import shutil
import tempfile

import gdown

from lipreader.paths import DATA_ROOT

GRID_S1_URL = "https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL"


def download_grid_s1(data_root: str = DATA_ROOT, url: str = GRID_S1_URL) -> None:
    os.makedirs(data_root, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(data_root))) as tmp:
        zip_path = os.path.join(tmp, "data.zip")
        gdown.download(url, zip_path, quiet=False)
        print("Extracting...")
        gdown.extractall(zip_path, to=tmp)
        extracted = os.path.join(tmp, "data")
        for name in os.listdir(extracted):
            if name == "Thumbs.db":
                continue
            src, dst = os.path.join(extracted, name), os.path.join(data_root, name)
            if os.path.exists(dst):
                print(f"  {dst} already exists, skipping")
                continue
            shutil.move(src, dst)
    print(f"GRID s1 ready in {data_root}")


if __name__ == "__main__":
    download_grid_s1()
