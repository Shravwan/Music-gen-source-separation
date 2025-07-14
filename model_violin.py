import torch.nn as nn

class ViolinClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),  # (128 → 63)
            nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2), # (63 → 30)
            nn.Flatten(),
            nn.Linear(32 * 30 * 30, 64), nn.ReLU(),
            nn.Linear(64, 1)  # Output: Binary class
        )

    def forward(self, x):
        return self.net(x)
