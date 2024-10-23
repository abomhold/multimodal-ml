from image.image_preprocessing import preprocess_image
from image.image_utils import get_classes, get_image, create_dataset, split_train_val_dataset, match_userid_image, process_dataframe
from image.image_model import get_pretrained_vgg16
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import itertools
import pandas as pd


def save_model_weights(model, path="vgg16_weights.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


def load_pretrained_model(num_classes=2, path="vgg16_weights.pth", device="cpu"):
    model = get_pretrained_vgg16(num_classes=num_classes, freeze_features=False)

    model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)


def train(train_image_path, train_class_path, device, num_epochs=5, optimizer_choice='adam'):
    train_images = preprocess_image(train_image_path)
    train_classes = get_classes(train_class_path)

    dataset = create_dataset(train_images, train_classes)
    train_loader, val_loader = split_train_val_dataset(dataset)

    model = get_pretrained_vgg16().to(device)

    if optimizer_choice == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    step_interval = 10

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        step_count = 0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss
            step_count += 1

            if (step + 1) % step_interval == 0:
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == labels).sum().item() / labels.size(0)
                print(
                    f"Ecpoch [{epoch + 1}], Step[{step + 1}], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%")

        scheduler.step()

        avg_loss = total_loss / step_count
        print(f'Epoch[{epoch + 1}] completed. Average Loss: {avg_loss:.4f}')

        validate(model, val_loader, criterion, device)

    save_model_weights(model, path='vgg16_weights.pth')


def validate(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_val_loss += loss
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct / total
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")
    model.train()


def test(test_image_path, dataframe, device):
    test_image = get_image(test_image_path)
    test_classes = process_dataframe(dataframe)

    matched_dict = match_userid_image(test_image, test_classes)

    model = load_pretrained_model(num_classes=2, path='vgg16_weights.pth', device=device)

    model.eval()
    with torch.no_grad():
        for user_id, (image, label) in matched_dict.items():
            image = image.to(device)

            image = image.unsqueeze(0)

            output = model(image)
            output = F.sigmoid(output)
            dataframe.loc[dataframe['userid'] == user_id, 'gender'] = output.item()

    return dataframe
