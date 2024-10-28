import ssl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Bypass errors
ssl._create_default_https_context = ssl._create_unverified_context

# Get dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
testset = torchvision.datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

loaders = {
    'train': torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True),
    'test': torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
}

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
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training Loop
def train_model():
    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(loaders['train'], 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
            optimizer.zero_grad()
            outputs = model(inputs)  # Use the instantiated model
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # Save the trained model
    torch.save(model.state_dict(), 'emnist_model.pth')

# Train the model and save it
train_model()