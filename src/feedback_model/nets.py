"""NightLightsNet is the CNN for VIIRS Vessel Detection and continuous retraining
"""
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, flatten

# N_CHANNELS = 4
# N_CHANNELS = 2  # this should be set from dataset
OUT_CHANNELS = 20
KERNEL_SIZE = 5
FC_FEATURES_IN = 120
FC_FEATURES_OUT = 256
N_CLASSES = 2
DROPOUT_RATE = 0.5
INPUT_SIZE = OUT_CHANNELS


class NightLightsNet(nn.Module):
    """CNN with 4 channels for moonlight, clouds, nanowatts and the land-sea."""

    def __init__(self, N_CHANNELS: int):
        """ """
        super().__init__()
        self.conv1 = nn.Conv2d(N_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(OUT_CHANNELS, OUT_CHANNELS * 2, KERNEL_SIZE)

        # Calculating size after conv and pooling operations
        conv1_out_size = INPUT_SIZE - KERNEL_SIZE + 1
        pooled_out_size = conv1_out_size // 2
        conv2_out_size = pooled_out_size - KERNEL_SIZE + 1
        pooled_out_size2 = conv2_out_size // 2

        linear_input = pooled_out_size2 * pooled_out_size2 * OUT_CHANNELS * 2

        self.fc1 = nn.Linear(linear_input, FC_FEATURES_IN)
        self.fc2 = nn.Linear(FC_FEATURES_IN, FC_FEATURES_OUT)
        self.fc3 = nn.Linear(FC_FEATURES_OUT, N_CLASSES)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Parameters
        ----------
        x : Tensor

        Returns
        -------
        Tensor

        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
