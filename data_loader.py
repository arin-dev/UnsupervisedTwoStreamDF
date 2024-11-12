# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import dlib
import sys
from lib.vaf_util import get_crops_landmarks

class VideoDataset(Dataset):
    def __init__(self, frame_direc, face_detector_path, transform=None):
        self.frame_direc = frame_direc
        self.frames = sorted(os.listdir(frame_direc))
        self.transform = transform
        self.face_detector, self.sp68 = self.load_face_detector(face_detector_path)

    def load_face_detector(self, face_detector_path):
        if not os.path.isfile(face_detector_path):
            print("Could not find shape_predictor_68_face_landmarks.dat")
            sys.exit()
        face_detector = dlib.get_frontal_face_detector()
        sp68 = dlib.shape_predictor(face_detector_path)
        return face_detector, sp68

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # Load frame
        num_frames = 12 # FOR NOW
        cropped_frames = []

        for i in range(num_frames):
            frame_path = os.path.join(self.frame_direc, self.frames[idx + i])  # Adjust index for multiple frames
            frame = Image.open(frame_path) #.convert('RGB')
            frame = frame.convert('RGB') 
            frame_np = np.array(frame)

            # Debugging checks
            print(f"Frame shape: {frame_np.shape}")  # Should be (height, width, 3)
            print(f"Frame dtype: {frame_np.dtype}")  # Should be uint8
            print(f"Pixel value range: {frame_np.min()} to {frame_np.max()}")  # Should be 0 to 255
            
            # Perform face detection and extract features for each frame
            print(f"Frame shape: {frame_np.shape} direc: {frame_path} and len till now: {len(cropped_frames)}")  # Should be (height, width, 3) for RGB
            face_crops, landmarks = get_crops_landmarks(self.face_detector, self.sp68, frame_np)

            if len(face_crops) == 0:
                print(f"No face detected in {frame_path}. Skipping this frame.")
                return None  # Skip frames with no detected face

            cropped_frames.append(face_crops[0])  # Append the first crop if found

        return torch.stack(cropped_frames)  # Return stacked frames as a tensor
    

def get_data_loaders(frame_direc, face_detector_path, batch_size=1):  # Batch size is set to 1 for now but needs to be changes along with how get_load_loaders is passing data to its caller.
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = VideoDataset(frame_direc, face_detector_path, transform) # CROPPED DATA
    # return dataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)