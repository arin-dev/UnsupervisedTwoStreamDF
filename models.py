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
    def __init__(self, input_channels=3, num_classes=128, frames_per_clip=12, height=128, width=128): # Add parameters
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

class TwoStreamNetworkTransferLearning(nn.Module):
    def __init__(self, spatial_weight=0.5, temporal_weight=0.5):
        super(TwoStreamNetworkTransferLearning, self).__init__()
        self.spatial_stream = SpatialStreamResNet()
        self.temporal_stream = TemporalStream3DResNet()
        self.fc_final = nn.Linear(128 * 2, 1)

    def forward(self, frames):
        spatial_predictions = []
        for frame in frames:
            spatial_output = self.spatial_stream(frame.unsqueeze(0)) # Add batch dimension
            spatial_predictions.append(spatial_output)
        
        spatial_features = torch.mean(torch.stack(spatial_predictions), dim=0)
        # Reshape for temporal stream: (Batch, Channels, Frames, Height, Width)
        # temporal_input = frames.permute(0, 2, 1, 3, 4)  # Swap Channels (dim 1) and Frames (dim 2)
        frames = frames.unsqueeze(0)
        temporal_input = frames.permute(0, 2, 1, 3, 4)  # Swap Channels (dim 1) and Frames (dim 2)

        # temporal_input = frames.permute(1, 0, 2, 3)  # Swap Channels (dim 1) and Frames (dim 2)

        temporal_features = self.temporal_stream(temporal_input)

        combined_features = torch.cat((spatial_features, temporal_features), dim=1)
        output = torch.sigmoid(self.fc_final(combined_features))
        return output