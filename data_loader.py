# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import dlib
import sys
import cv2
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
        frame_path = os.path.join(self.frame_direc, self.frames[idx])
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Could not open image file: {frame_path}")
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_crops, _ = get_crops_landmarks(self.face_detector, self.sp68, img)

        if len(face_crops) == 0:
            print(f"No face detected in {frame_path}. Skipping this frame.")
            return None

        return face_crops[0]  # Return the first crop directly
    

def get_data_loaders(frame_direc, face_detector_path, batch_size=1):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = VideoDataset(frame_direc, face_detector_path, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)