from Data_loader import *

# CNN Model Definition
class CNN_3(nn.Module):
    def __init__(self):
        super(CNN_3, self).__init__()
        

        # Convolution Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 

        # Pooling Layers 
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    # Forward Definition
    def forward(self, x):
        
        x = self.pool(torch.relu(self.conv1(x)))  
        x = self.pool(torch.relu(self.conv2(x)))  
        x = torch.relu(self.conv3(x))             

        # Flatten of the data from 2-D to 1-D 
        x = x.view(-1, 128 * 7 * 7)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x
