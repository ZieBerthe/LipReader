import os
import cv2
import numpy as np
import torch 
from typing import List, Tuple
from matplotlib import pyplot as plt
import imageio
import gdown

torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#build our data loading fonctions.
 
def download_data(url: str, output_path: str) -> None:
    """Download data from a given URL and save it to the specified output path."""
    gdown.download(url, output_path, quiet=False)
    print("zip file downloaded, now extracting...")
    gdown.extractall(output_path)
    print(f"Data downloaded and extracted to {output_path}")
if __name__ == "__main__":
    url = "https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL"
    output_path = "data.zip"
    download_data(url, output_path)
