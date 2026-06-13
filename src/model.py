import torch.nn as nn

class LandCNN(nn.Module):
    def __init__(self, in_channels=13, num_classes=10):
        super().__init__()

    
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),  # 1
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),           # 2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),           # 3
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),           # 4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),          # 5
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),         # 6
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 🔥 Classification Head
        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 8 * 8, 512),               # 7
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),                       # 8
            nn.ReLU(),

            nn.Linear(256, num_classes)                # 9
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x