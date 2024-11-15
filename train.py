import torch
import torch.optim as optim
import torch.nn as nn
from models import TwoStreamNetworkTransferLearning
from data_loader import get_data_loaders
import json
import os

def train_model(num_epochs, frame_direc, face_detector_path, device):
    print("Entering to train data!")
    model = TwoStreamNetworkTransferLearning().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    print("Opening Label.json !")
    with open('labels.json', 'r') as file:
        label_map = json.load(file)
    
    model.train()
    for epoch in range(num_epochs):
        for video_folder in os.listdir(frame_direc):
            video_name = '_'.join(video_folder.split('_')[:-1])
            
            if not video_name.endswith('.mp4'):
                print(f"Folder seems invalid : {video_name}! Skipping to next.")
                continue
            
            train_loader = get_data_loaders(os.path.join(frame_direc, video_folder), face_detector_path, batch_size=1)
            data_to_model = []
            i = 0
            for data in train_loader:
                if data is None:
                    continue

                labels = label_map.get(video_name)
                if labels is None:
                    print(f"Label for {video_name} not found! Skipping to next.")
                    continue
                
                i += 1
                # print(f"CURRENTLY {i}th time IN TRAIN: {data.shape}")
                data_to_model.append(data)
            print(f"CURRENTLY training with {video_name} / {video_folder} : {label_map[video_name]}")

            optimizer.zero_grad()
            data_to_model = torch.stack(data_to_model, dim=0).to(device)
            for batch in data_to_model:
                # print(batch.shape)
                outputs = model(batch)
                print(outputs)
                # labels = torch.tensor(labels).to(device)
                labels = torch.tensor([label_map[video_name]]).float().to(device).unsqueeze(1) # Add dimension and convert to float
                # print(labels.shape)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    torch.save(model.state_dict(), 'two_stream_model.pth')


if __name__ == "__main__":
    num_epochs = 1
    frame_direc = 'frames_function_test_data'
    face_detector_path = 'shape_predictor_68_face_landmarks.dat'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_model(num_epochs, frame_direc, face_detector_path, device)