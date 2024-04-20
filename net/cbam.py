import torch
import torch.nn as nn


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()

        # Calculate intermediate channels for efficient bottleneck
        mid_channels = in_channels // reduction_ratio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling layer
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling layer

        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels),  # Reduce dimensionality
            nn.ReLU(),  # Non-linear activation
            nn.Linear(mid_channels, in_channels)  # Restore dimensionality
        )
        self.sigmoid = nn.Sigmoid()  # Sigmoid for smooth attention weights

    def forward(self, x):
        # Average pooling and squeeze channels
        avg_out = self.avg_pool(x).view(x.size(0), -1)  
        # Max pooling and squeeze channels
        max_out = self.max_pool(x).view(x.size(0), -1)

        # Process with shared MLP layers
        avg_out = self.shared_mlp(avg_out)
        max_out = self.shared_mlp(max_out)

        # Combine attention, apply sigmoid, and reshape back to feature map size
        output = self.sigmoid(avg_out + max_out)
        return output.unsqueeze(2).unsqueeze(3) * x  # Multiply with input


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv2d = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Max pooling across channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate average and max pooled features
        output = torch.cat([avg_out, max_out], dim=1)

        # Convolution for spatial attention
        output = self.conv2d(output)
        # Apply sigmoid and multiply with original input
        return self.sigmoid(output) * x 


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttentionModule(kernel_size)

    def forward(self, x):
        # Apply channel attention first
        out = self.channel_attention(x)
        # Then apply spatial attention 
        out = self.spatial_attention(out)
        return out 