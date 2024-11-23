import torch
import torch.optim as optim
import torch.nn as nn
from models import TwoStreamNetworkTransferLearning
from data_loader import get_data_loaders
import json
import os

def train_model(num_epochs, frame_direc, face_detector_path, device, batch_size=1):
    print("Entering to train data!")
    model = TwoStreamNetworkTransferLearning().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    print("Opening Label.json!")
    with open('labels.json', 'r') as file:
        label_map = json.load(file)

    model.train()
    for epoch in range(num_epochs):
        # for video_folder in os.listdir(frame_direc):
            # video_name = '_'.join(video_folder.split('_')[:-1])

            # print(video_folder)

            # if not video_name.endswith('.mp4'):
            #     print(f"Folder seems invalid: {video_name}! Skipping to next.")
            #     continue

        # print(os.path.join(frame_direc, video_folder))
        train_loader = get_data_loaders(frame_direc, face_detector_path, batch_size=batch_size)
        
        # print(f"CURRENTLY training with {video_name} / {video_folder} : {label_map.get(video_name)}")
        batch_count = 0
        for data, video_names in train_loader:
            print(f"Currently working with : {video_names}")
            print(data.shape)
            if data is None:
                print("No valid data returned from loader, skipping this batch.")
                continue

            # Fetch labels and ensure they're the right shape
            # print(data.shape)

            # labels = label_map.get(video_name)
            # if labels is None:
            #     print(f"Label for {video_name} not found! Skipping to next.")
            #     continue
            
            # Here we assume labels is a single value for binary classification
            # Repeat the label for the batch size and ensure it's the right shape
            # labels_tensor = torch.tensor([labels] * data.size(0)).float().to(device).unsqueeze(1)  # Shape: [batch_size, 1]
            # print(f"Training on batch of size {data.size(0)} with labels: {labels_tensor.squeeze().tolist()}")

            data = data.to(device)  # Move data to the appropriate device

            optimizer.zero_grad()
            outputs = model(data)
            outputs = outputs.squeeze(1)
            # print("OUTPUT TYPE", type(outputs))


            labels_tensor = []
            for video_name in video_names:
                labels_tensor.append(label_map.get(video_name))
            
            labels_tensor = torch.tensor(labels_tensor).float().to(device)
            # print("label TYPE", type(labels_tensor), labels_tensor)

            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()

            print(f'Batch {batch_count+1}/{len(train_loader)}, Loss: {loss.item()}')
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        
    torch.save(model.state_dict(), 'two_stream_model.pth')

if __name__ == "__main__":
    num_epochs = 1
    frame_direc = 'frames_function_test_data'
    face_detector_path = 'shape_predictor_68_face_landmarks.dat'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_model(num_epochs, frame_direc, face_detector_path, device, 5)
