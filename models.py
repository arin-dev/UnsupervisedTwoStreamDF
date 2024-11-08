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
    def __init__(self):
        super(TwoStreamNetworkTransferLearning, self).__init__()
        self.spatial_stream = SpatialStreamResNet()
        self.temporal_stream = TemporalStream3DResNet()
        self.fc_final = nn.Linear(128 * 2, 1)
    
    def forward(self, rgb_frame, optical_flow_stack):
        spatial_features = self.spatial_stream(rgb_frame)
        temporal_features = self.temporal_stream(optical_flow_stack)
        combined_features = torch.cat((spatial_features, temporal_features), dim=1)
        output = torch.sigmoid(self.fc_final(combined_features))
        return output
