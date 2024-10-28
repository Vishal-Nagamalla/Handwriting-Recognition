import ssl
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Define transforms for the dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Load the dataset
train_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='data', train=False, transform=transform, download=True)

# Create data loaders
loaders = {
    'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2),
    'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)
}

# Load the pre-trained ViT model and processor
model_name = "google/vit-base-patch16-224-in21k"
image_processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, num_labels=10)

# Set device to CPU
device = torch.device('cpu')
model.to(device)

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# Training function
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data).logits
        loss = loss_fn(outputs, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\tLoss: {loss.item():.6f}')

# Testing function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            outputs = model(data).logits
            test_loss += loss_fn(outputs, target).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders['test'].dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders["test"].dataset)} ({100. * correct / len(loaders["test"].dataset):.0f}%)\n')

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)
    image = image.resize((224, 224))
    image = transforms.Grayscale(num_output_channels=3)(image)
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    output = model(image).logits
    prediction = output.argmax(dim=1, keepdim=True).item()
    print(f'Prediction: {prediction}')
    plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.show()

# Main script
if __name__ == '__main__':
    for epoch in range(1, 6):  # Reduced number of epochs for faster training
        train(epoch)
        test()

    # Test the model with a custom image
    predict_image('/Users/vishal/Documents/Programming/Code/Internship/Atyeti/NN_HandwritingRecognition/images/numbers/3_4.png')