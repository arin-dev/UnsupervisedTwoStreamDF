# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import dlib
import sys
import cv2
from sklearn.cluster import KMeans
from lib.vaf_util import get_crops_landmarks, extract_features_eyes, extract_features_mouth

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
        return len(self.rgb_frames)

    def __getitem__(self, idx):
        # Load RGB frame
        rgb_frame_path = os.path.join(self.rgb_dir, self.rgb_frames[idx])
        rgb_frame = Image.open(rgb_frame_path).convert('RGB')
        rgb_frame_np = np.array(rgb_frame)
        
        # Perform face detection and extract features
        face_crops, landmarks = get_crops_landmarks(self.face_detector, self.sp68, rgb_frame_np)
        
        if len(landmarks) == 0:
            print(f"No face detected in {rgb_frame_path}. Skipping this set!!")
            return None  # Skip frames with no detected face
        
        face_crop = face_crops[0]
        landmarks = landmarks[0]
        
        # Extract eyes and mouth features
        features_eyes = extract_features_eyes(landmarks, face_crop)
        features_mouth = extract_features_mouth(landmarks, face_crop)
        
        if features_eyes is None or features_mouth is None:
            print(f"Feature extraction failed for {rgb_frame_path}. Skipping frame.")
            return None

        spatial_features = np.concatenate((features_eyes, features_mouth))
        spatial_features = torch.tensor(spatial_features, dtype=torch.float32)

        # Load and stack optical flow frames, ensuring idx + 10 does not exceed limits
        flow_stack = []
        for i in range(idx, min(idx + 10, len(self.flow_frames))):
            flow_path = os.path.join(self.flow_dir, self.flow_frames[i])
            flow_frame = Image.open(flow_path).convert('L')
            flow_stack.append(np.array(flow_frame))

        flow_stack = np.stack(flow_stack, axis=0)
        flow_stack = torch.tensor(flow_stack, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            rgb_frame = self.transform(rgb_frame)

        return spatial_features, flow_stack

def get_data_loaders(rgb_dir, flow_dir, face_detector_path, batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = VideoDataset(rgb_dir, flow_dir, face_detector_path, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)