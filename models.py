import torch
import torch.nn as nn
import torchvision.models as models

class SpatialStreamResNet(nn.Module):
    def __init__(self):
        super(SpatialStreamResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 128)
    
    def forward(self, x):
        x = self.resnet_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TemporalStream3DResNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=128, frames_per_clip=12, height=128, width=128):
        super(TemporalStream3DResNet, self).__init__()
        self.resnet3d = models.video.r3d_18(weights='DEFAULT')
        self.resnet3d.stem[0] = nn.Conv3d(input_channels, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3))
        self.resnet3d = nn.Sequential(*list(self.resnet3d.children())[:-1])

        # Use correct dimensions for dummy input
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, frames_per_clip, height, width)
            feature_map = self.resnet3d(dummy_input)

        self.fc = nn.Linear(feature_map.numel(), num_classes)

    def forward(self, x):
        x = self.resnet3d(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# class TwoStreamNetworkTransferLearning(nn.Module):
#     def __init__(self, spatial_weight=0.5, temporal_weight=0.5):
#         super(TwoStreamNetworkTransferLearning, self).__init__()
#         self.spatial_stream = SpatialStreamResNet()
#         self.temporal_stream = TemporalStream3DResNet()
#         self.fc_final = nn.Linear(128 * 2, 1)

#     def forward(self, frames):
#         # Process the entire batch for spatial stream
#         spatial_output = self.spatial_stream(frames)  # frames already has batch dimension
#         spatial_features = torch.mean(spatial_output, dim=0)  # Average over the batch

#         # Prepare input for temporal stream
#         temporal_input = frames.permute(0, 2, 1, 3, 4)  # Swap Channels (dim 1) and Frames (dim 2)

#         temporal_features = self.temporal_stream(temporal_input)

#         combined_features = torch.cat((spatial_features, temporal_features), dim=1)
#         output = torch.sigmoid(self.fc_final(combined_features))
#         return output
class TwoStreamNetworkTransferLearning(nn.Module):
    def __init__(self, spatial_weight=0.5, temporal_weight=0.5):
        super(TwoStreamNetworkTransferLearning, self).__init__()
        self.spatial_stream = SpatialStreamResNet()
        self.temporal_stream = TemporalStream3DResNet()
        self.fc_final = nn.Linear(128 * 2, 1)  # Final output layer

    def forward(self, frames):
        # frames should have shape [batch_size, 12, 3, 128, 128]
        batch_size, frames_no, channels, height, width = frames.shape

        # Process the entire batch for the spatial stream
        # Reshape frames for spatial processing: average over the frames dimension
        spatial_input = frames.view(-1, channels, height, width)  # Shape: [batch_size * frames_no, channels, height, width]
        spatial_output = self.spatial_stream(spatial_input)  # Output will have shape [batch_size * frames_no, num_features]

        # Average spatial features over frames: (batch_size, frames_no, features)
        spatial_features = spatial_output.view(batch_size, frames_no, -1).mean(dim=1)  # Shape: [batch_size, num_features]

        # Prepare input for the temporal stream (no change needed as input is in correct format)
        # temporal_input = frames  # Shape: [batch_size, frames_no, channels, height, width]
        temporal_input = frames.permute(0, 2, 1, 3, 4)  # Swap Channels (dim 1) and Frames (dim 2)


        # Process the temporal stream
        temporal_features = self.temporal_stream(temporal_input)  # Shape: [batch_size, num_classes]

        # Combine the features from both streams
        combined_features = torch.cat((spatial_features, temporal_features), dim=1)  # Concatenate along feature dimension
        output = torch.sigmoid(self.fc_final(combined_features))  # Final output layer
        return output