import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=2, padding=1),  # Input: [B, 3, T, H, W]
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output pixel values between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# DataLoader function (assuming you have implemented get_data_loaders)
def get_data_loaders(frame_direc, face_detector_path, batch_size=1):
    """
    Placeholder for your existing data loader function.
    Should return a DataLoader that yields (data, video_names).
    """
    # TODO: Implement your data loading mechanism here.
    pass

def train_autoencoder(num_epochs, frame_direc, face_detector_path, device, batch_size=1):
    print("Starting autoencoder training...")
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # For unsupervised learning, we don't need labels during training.
    model.train()
    for epoch in range(num_epochs):
        train_loader = get_data_loaders(frame_direc, face_detector_path, batch_size=batch_size)
        batch_count = 0
        epoch_loss = 0.0

        for data, video_names in train_loader:
            if data is None:
                print("No valid data returned from loader, skipping this batch.")
                continue

            data = data.to(device)  # Move data to the appropriate device

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)  # Compare output to input
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_count+1}], Loss: {loss.item():.6f}')
            batch_count += 1

        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.6f}')

    torch.save(model.state_dict(), 'autoencoder.pth')
    print("Autoencoder training completed.")

def compute_reconstruction_errors(frame_direc, face_detector_path, device, batch_size=1):
    print("Computing reconstruction errors...")
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load('autoencoder.pth', map_location=device))
    model.eval()

    errors = {}
    with torch.no_grad():
        data_loader = get_data_loaders(frame_direc, face_detector_path, batch_size=batch_size)
        for data, video_names in data_loader:
            if data is None:
                print("No valid data returned from loader, skipping this batch.")
                continue

            data = data.to(device)
            outputs = model(data)
            loss = ((outputs - data) ** 2).mean(dim=[1, 2, 3, 4])  # Compute MSE per sample
            loss = loss.cpu().numpy()

            for idx, video_name in enumerate(video_names):
                errors[video_name] = loss[idx]
                print(f'Computed error for video {video_name}: {loss[idx]:.6f}')

    return errors

def assign_labels(errors, threshold=None):
    if threshold is None:
        # Set threshold based on the error distribution
        # For example, mean + k * std deviation
        error_values = list(errors.values())
        mean_error = np.mean(error_values)
        std_error = np.std(error_values)
        threshold = mean_error + std_error  # You can adjust the multiplier

    labels = {}
    for video_name, error in errors.items():
        # Assign labels based on reconstruction error
        labels[video_name] = 0 if error <= threshold else 1  # 0 for real, 1 for deepfake
        print(f'Video {video_name}: Error={error:.6f}, Label={labels[video_name]}')

    # Save labels to a JSON file
    with open('unsupervised_labels.json', 'w') as f:
        json.dump(labels, f)

    return labels

if __name__ == "__main__":
    num_epochs = 10
    frame_direc = 'frames_function_test_data'  # Update with your directory
    face_detector_path = 'shape_predictor_68_face_landmarks.dat'  # Update if needed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 5  # Adjust as needed

    # Step 1: Train the autoencoder
    train_autoencoder(num_epochs, frame_direc, face_detector_path, device, batch_size)

    # Step 2: Compute reconstruction errors for each video
    errors = compute_reconstruction_errors(frame_direc, face_detector_path, device, batch_size)

    # Step 3: Assign labels based on errors
    labels = assign_labels(errors)

    print("Unsupervised labeling completed.")