from Data_loader import *

# CNN Model Definition
class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        
        # Convolution Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        # Pooling Layers
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Activation Function Layers
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
       
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    # Forward Definition
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten of the data from 2-D to 1-D 
        x = x.view(-1, 64 * 6 * 6)

        x = self.fc1(x)
        x= torch.relu(x)
        x = self.fc2(x)

        return x
