# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from models import TwoStreamNetworkTransferLearning
from data_loader import get_data_loaders

import json
import os

def train_model(num_epochs, frame_direc, face_detector_path, device):
    model = TwoStreamNetworkTransferLearning().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    with open('labels.json', 'r') as file:
        label_map = json.load(file)
    
    model.train()
    for epoch in range(num_epochs):
        for video_folder in os.listdir(frame_direc):
            if not video_folder.endswith('.mp4'):
                continue  # Skip non-video folders
            
            video_name = '_'.join(video_folder.split('_')[:-1])  # Get video_name till the last '_'
            train_loader = get_data_loaders(os.path.join(frame_direc, video_folder), face_detector_path)

            for data in train_loader:
                if data is None:
                    continue  # Skip if data is None
                spatial_features, flow_stacks = data
                spatial_features, flow_stacks = spatial_features.to(device), flow_stacks.to(device)

                optimizer.zero_grad()
                outputs = model(spatial_features, flow_stacks)

                labels = label_map.get(video_name)
                if labels is None:
                    print(f"Label for {video_name} not found! Skipping to next.")
                    continue
                labels = torch.tensor(labels).to(device)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    torch.save(model.state_dict(), 'two_stream_model.pth')
