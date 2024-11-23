import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import dlib
import sys
import cv2
from lib.vaf_util import get_crops_landmarks
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, frame_direc, face_detector_path, transform=None):
        self.frame_direc = frame_direc
        self.subfolders = sorted(os.listdir(frame_direc))  # List of subfolders
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
        return len(self.subfolders)  # Each subfolder is one data point

    def __getitem__(self, idx):
        batch_images = []
        video_names = []

        # Get the path of the subfolder
        subfolder_path = os.path.join(self.frame_direc, self.subfolders[idx])
        frame_files = sorted(os.listdir(subfolder_path))  # List all frames in the subfolder
        # print(f"Loading frames from: {subfolder_path} with {len(frame_files)} frames.")

        # Load exactly 12 frames
        for i in range(12):  # Assumes each subfolder has exactly 12 frames
            frame_path = os.path.join(subfolder_path, frame_files[i])
            if not frame_path.endswith('.jpg'):
                print(f"Invalid file type: {frame_path}. Skipping to next.")
                continue
            
            img = cv2.imread(frame_path)
            if img is None:
                print(f"Could not open image file: {frame_path}")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_crops, _ = get_crops_landmarks(self.face_detector, self.sp68, img)

            if len(face_crops) == 0:
                print(f"No face detected in {frame_path}. Skipping this frame.")
                continue

            face_crop = Image.fromarray(face_crops[0])
            if self.transform:
                # print("Applied transform!")
                face_crop = self.transform(face_crop)

            # Ensure the face crop is a tensor and has the expected shape
            if face_crop.shape[0] != 3 or face_crop.shape[1] != 128 or face_crop.shape[2] != 128:
                print(f"Face crop shape is not valid: {face_crop.shape}. Skipping this frame.")
                continue

            # print(f"Adding face crop of shape: {face_crop.shape} to batch.")
            batch_images.append(face_crop)  # Append the transformed face crop

        # Ensure exactly 12 frames are returned
        if len(batch_images) < 12:
            print(f"Insufficient frames collected for index {idx}. Expected 12, got {len(batch_images)}.")
            return None, None
            raise ValueError(f"Insufficient frames collected for batch. Expected 12, got {len(batch_images)}.")

        # label = label_map['_'.join(video_folder.split('_')[:-1])]
        video_name = '_'.join(self.subfolders[idx].split('_')[:-1])
        # label = label_map.get(video_name)
        # print(f"is this the flder? : {self.subfolders[idx]}, then vid name : {video_name}, label = {label}")
        # Return a tensor of shape [12, 3, 128, 128]
        # print(f"Returning a batch of shape: {len(batch_images)}, {[img.shape for img in batch_images]}")  # Debug shapes
        return torch.stack(batch_images), video_name  # Shape will be [12, 3, 128, 128]

def get_data_loaders(frame_direc, face_detector_path, batch_size=1):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = VideoDataset(frame_direc, face_detector_path, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)