from image.image_preprocessing import preprocess_image
from image.image_utils import get_classes, get_image, create_dataloader
from image.image_model import VGG16
import torch.optim as optim
import torch
import torch.nn as nn


def save_model_weights(model, path="vgg16_weights.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


def load_model_weights(model, path="vgg16_weights.pth", device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model weights loaded from {path}")

def train(train_image_path, train_class_path, device, num_epochs=50):
    train_images = preprocess_image(train_image_path)
    train_classes = get_classes(train_class_path)
    train_loader = create_dataloader(train_images, train_classes)

    model = VGG16(2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    step_interval = 10

    model.train()
    for epoch in num_epochs:
        total_loss = 0.0
        step_count = 0

        for step, (image, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backword()
            optimizer.step()

            total_loss += loss
            step_count += 1

            if (step + 1) % step_interval == 0:
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == labels).sum().item() / labels.size(0)
                print(f"Ecpoch [{epoch+1}], Step[{step+1}], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%")

        avg_loss = total_loss / step_count
        print(f'Epoch[{epoch+1}] completed. Average Loss: {avg_loss:.4f}')

    save_model_weights(model, path='vgg16_weights.pth')


def test(test_image_path, test_class_path, device):
    test_image = get_image(test_image_path)
    test_classes = get_classes(test_class_path)
    test_loader = create_dataloader(test_image, test_classes)

    model = VGG16(2).to(device)
    load_model_weights(model, path="vgg16_weights.pth", device=device)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += 1
            correct += (outputs == labels).sum().item()

        test_accuracy = correct / total
        print(f'Final Test Accuracy: {test_accuracy * 100:.2f}')

