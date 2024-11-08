# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from models import TwoStreamNetworkTransferLearning
from data_loader import get_data_loaders

import json


# def train_model(num_epochs, rgb_dir, flow_dir, face_detector_path, device):
def train_model(num_epochs, frame_direc, face_detector_path, device):
    model = TwoStreamNetworkTransferLearning().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    # train_loader = get_data_loaders(rgb_dir, flow_dir, face_detector_path)
    train_loader = get_data_loaders(frame_direc, face_detector_path)

    with open('labels.json', 'r') as file:
        label_map = json.load(file)
    
    model.train()
    for epoch in range(num_epochs):
        for data in train_loader:
            if data is None:
                continue  # Skip if data is None
            spatial_features, flow_stacks = data
            spatial_features, flow_stacks = spatial_features.to(device), flow_stacks.to(device)

            optimizer.zero_grad()
            outputs = model(spatial_features, flow_stacks)
            # labels = torch.ones(outputs.shape).to(device)  # Replace with actual labels
            label = label_map["file_name"]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    torch.save(model.state_dict(), 'two_stream_model.pth')
