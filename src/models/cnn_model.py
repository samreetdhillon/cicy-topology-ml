"""
CNN model architecture for classifying CICY manifolds
based on configuration matrices and an additional scalar feature
(e.g., ambient dimension).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CICYClassifier(nn.Module):
    """
    CNN with a shared backbone and two classification heads: h^{1,1} and h^{2,1}.
    """

    def __init__(self, num_h11_classes=20, num_h21_classes=102):
        super().__init__()

        # CNN backbone
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive pooling removes hard-coded spatial assumptions
        # Output shape: (B, 64, 3, 3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))

        # Fully connected shared layer
        # +1 for scalar feature (ambient dimension)
        self.fc_shared = nn.Linear(64 * 3 * 3 + 1, 256)

        # Output heads
        self.head_h11 = nn.Linear(256, num_h11_classes)
        self.head_h21 = nn.Linear(256, num_h21_classes)

    def forward(self, x_img, x_scalar):
        """
        Parameters:
        x_img : torch.Tensor
            Shape (B, 1, 12, 15) — CICY configuration matrices
        x_scalar : torch.Tensor
            Shape (B, 1) — scalar feature (ambient dimension)

        Returns Classification logits
        """

        # Sanity checks
        assert x_img.dim() == 4, "x_img must be (B, 1, H, W)"
        assert x_img.shape[-2:] == (12, 15), "Expected CICY matrix of size 12×15"
        assert x_scalar.dim() == 2 and x_scalar.shape[1] == 1, \
            "x_scalar must have shape (B, 1)"

        # CNN feature extraction
        x = self.pool(F.relu(self.conv1(x_img)))
        x = self.pool(F.relu(self.conv2(x)))

        # Spatial normalization
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Concatenate scalar feature
        x = torch.cat([x, x_scalar], dim=1)

        # Shared dense layer
        x = F.relu(self.fc_shared(x))

        # Heads
        out_h11 = self.head_h11(x)
        out_h21 = self.head_h21(x)

        return out_h11, out_h21
