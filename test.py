import torch
from models import TwoStreamNetworkTransferLearning
from data_loader import get_data_loaders
import json
import os

def test_model(frame_direc, face_detector_path, device):
    model = TwoStreamNetworkTransferLearning().to(device)
    model.load_state_dict(torch.load('two_stream_model.pth'))
    model.eval()

    print("Opening Label.json !")
    with open('labels.json', 'r') as file:
        label_map = json.load(file)

    correct = 0
    total = 0

    with torch.no_grad():
        for video_folder in os.listdir(frame_direc):
            video_name = '_'.join(video_folder.split('_')[:-1])
            if not video_name.endswith('.mp4'):
                continue
            
            test_loader = get_data_loaders(os.path.join(frame_direc, video_folder), face_detector_path, batch_size=1)
            data_to_model = []
            for data in test_loader:
                if data is None:
                    continue
                data_to_model.append(data)

            data_to_model = torch.stack(data_to_model, dim=0).to(device)

            for batch in data_to_model:
                outputs = model(batch)
                predicted = (outputs > 0.85).float() # change predicted condition as you increase epoch size
                labels = torch.tensor([label_map[video_name]]).float().to(device).unsqueeze(1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print(f'Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    frame_direc = 'frames_function_test_data'
    face_detector_path = 'shape_predictor_68_face_landmarks.dat'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_model(frame_direc, face_detector_path, device)
