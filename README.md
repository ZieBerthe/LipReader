# LipReader
reconstructing the words/ voice coming from a video of a person talking with no sond
# dependencies to install
opencv, matplotlib, gdown, pytorch, gdown, imageio

# for the moment
Shape of mouth frames: torch.Size([2, 74, 100, 200])
Batch size: 2
Feature dimensions (H, W): 100 200 a single pic
Data type of char indices: torch.int64
Shape of char indices: torch.Size([2, 28]), so 28, 
Data type of audio STFT magnitude: torch.float32
Shape of audio STFT magnitude: torch.Size([2, 298, 101])
Audio feature dimensions (A, F): 298 101