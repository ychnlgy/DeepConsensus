import torch

class Cnn(torch.nn.Module):
    "This is an arbitrary model. Input 32x32 or 64x64 images."
    
    def __init__(self, channels, classes, imagesize):
        super(Cnn, self).__init__()
        
        if imagesize == (32, 32):
            firstpool = torch.nn.Sequential()
        elif imagesize == (64, 64):
            firstpool = torch.nn.MaxPool2d(2)
        else:
            raise AssertionError
        
        self.net = torch.nn.ModuleList([
        
            # BLOCK 0
            torch.nn.Sequential(
                torch.nn.Conv2d(channels, 32, 5, padding=2),
                torch.nn.BatchNorm2d(32),
                torch.nn.LeakyReLU(),
                firstpool
            ),
            
            # BLOCK 1: 32 -> 16
            torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, 3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool2d(2)
            ),
            
            # BLOCK 2: 16 -> 16
            torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, 3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.LeakyReLU()
            ),
            
            # BLOCK 3: 16 -> 8
            torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, 3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool2d(2)
            ),
            
            # BLOCK 4: 8 -> 8
            torch.nn.Sequential(
                torch.nn.Conv2d(128, 128, 3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.LeakyReLU()
            ),
            
            # BLOCK 5: 8 -> 4
            torch.nn.Sequential(
                torch.nn.Conv2d(128, 256, 3, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool2d(2)
            ),
            
            # BLOCK 6: 4 -> 4
            torch.nn.Sequential(
                torch.nn.Conv2d(256, 256, 3, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.LeakyReLU()
            )
        ])
        
        self.avg = torch.nn.AvgPool2d(4)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.2),
            
            torch.nn.Linear(1024, classes)
        )
    
    def forward(self, X):
        for layer in self.net:
            X = layer(X)
        avg = self.avg(X).view(-1, 256)
        return self.fc(avg)
