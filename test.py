import torch
from models import TwoStreamNetworkTransferLearning
from data_loader import get_data_loaders
import json
import os

def test_model(frame_direc, face_detector_path, device, batch_size):
    model = TwoStreamNetworkTransferLearning().to(device)
    model.load_state_dict(torch.load('two_stream_model.pth'))
    model.eval()

    print("Opening Label.json !")
    with open('labels_flipped_titles.json', 'r') as file:
        label_map = json.load(file)

    correct = 0
    total = 0

    test_loader = get_data_loaders(frame_direc, face_detector_path, batch_size)
    with torch.no_grad():
        # for video_folder in os.listdir(frame_direc):
        for data, video_names in test_loader:
            # video_name = '_'.join(video_folder.split('_')[:-1])
            # if not video_name.endswith('.mp4'):
            #     continue
            print(f"Currently working with : {video_names}")
            data = data.to(device)
            outputs = model(data)
            outputs = outputs.squeeze(1)
            # predicted = (outputs > 0.85).float() # change predicted condition as you increase epoch size
            labels_tensor = []
            for video_name in video_names:
                labels_tensor.append(label_map.get(video_name))

            labels_tensor = torch.tensor(labels_tensor).float().to(device)

            total += labels_tensor.size(0)
            # correct += (predicted == labels).sum().item()
            print(outputs, labels_tensor)

    # accuracy = correct / total * 100
    # print(f'Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    frame_direc = 'frames_function_test_data'
    face_detector_path = 'shape_predictor_68_face_landmarks.dat'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_model(frame_direc, face_detector_path, device, 5)
