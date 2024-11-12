# models.py
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
    def __init__(self, input_channels=20):
        super(TemporalStream3DResNet, self).__init__()
        self.resnet3d = models.video.r3d_18(pretrained=True)
        self.resnet3d.stem[0] = nn.Conv3d(input_channels, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3))
        self.resnet3d = nn.Sequential(*list(self.resnet3d.children())[:-1])
        self.fc = nn.Linear(self.resnet3d[-1][-1].in_features, 128)
    
    def forward(self, x):
        x = self.resnet3d(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TwoStreamNetworkTransferLearning(nn.Module):
    def __init__(self, spatial_weight=0.5, temporal_weight=0.5):
        super(TwoStreamNetworkTransferLearning, self).__init__()
        self.spatial_stream = SpatialStreamResNet()
        self.temporal_stream = TemporalStream3DResNet()

        # # WEIGHTED AVERAGE VARIABLES:
        # self.spatial_weight = spatial_weight
        # self.temporal_weight = temporal_weight

        # FOR USING FINAL LAYER
        self.fc_final = nn.Linear(128 * 2, 1)  # Define fc_final to combine both streams


    # # USING WEIGHTED AVERAGE TECHNIQUE:
    # def forward(self, frames):
    #     # Process each frame individually for spatial stream
    #     spatial_predictions = []
    #     for i in range(frames.size(1)):  # Loop through 12 frames
    #         spatial_output = self.spatial_stream(frames[:, i, :, :, :])  # Shape: (batch_size, 128)
    #         spatial_predictions.append(spatial_output)


    #     # Average spatial predictions
    #     spatial_features = torch.mean(torch.stack(spatial_predictions), dim=0)  # Shape: (batch_size, 128)
    #     spatial_prediction = torch.sigmoid(spatial_features)  # Apply sigmoid to averaged features

    #     # Process all frames together for temporal stream
    #     temporal_features = self.temporal_stream(frames)  # Shape: (batch_size, 128)
    #     temporal_prediction = torch.sigmoid(temporal_features)  # Apply sigmoid to temporal features

    #     # Weighted average of predictions
    #     final_prediction = (self.spatial_weight * spatial_prediction + self.temporal_weight * temporal_prediction) / \
    #                     (self.spatial_weight + self.temporal_weight)

    #     return final_prediction 



    # FOR USING FINAL LAYER
    def forward(self, frames):
        # Process each frame individually for spatial stream
        spatial_predictions = []
        for i in range(frames.size(1)):  # Loop through 12 frames
            spatial_output = self.spatial_stream(frames[:, i, :, :, :])  # Shape: (batch_size, 128)
            spatial_predictions.append(spatial_output)
        
        # Average spatial predictions
        spatial_features = torch.mean(torch.stack(spatial_predictions), dim=0)  # Shape: (batch_size, 128)

        # Process all frames together for temporal stream
        temporal_features = self.temporal_stream(frames)  # Shape: (batch_size, 128)

        # Concatenate spatial and temporal features
        combined_features = torch.cat((spatial_features, temporal_features), dim=1)  # Shape: (batch_size, 256)

        # Pass combined features through fc_final
        output = torch.sigmoid(self.fc_final(combined_features))  # Final prediction

        return output
