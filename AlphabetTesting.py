import ssl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Bypass SSL errors
ssl._create_default_https_context = ssl._create_unverified_context

# Create Models
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, 27)  # 26 letters + 1 (for indexing starting from 0)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = nn.MaxPool2d(2)(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Checking to see if GPU can be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model.load_state_dict(torch.load('emnist_model.pth'))
model.eval()  # Set the model to evaluation mode

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to predict the letter from an image
def predict_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    return chr(predicted.item() + 96)  # Convert to alphabet letter ('a' is 97 in ASCII)

# Test with an uploaded image
image_path = '/Users/vishal/Documents/Programming/Code/Internship/Atyeti/NN_HandwritingRecognition/images/letters/a.png'
predicted_letter = predict_image(image_path)
print(f'The predicted letter is: {predicted_letter}')