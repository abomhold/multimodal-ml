from image.image_preprocessing import preprocess_image
from image.image_utils import get_classes, get_image, create_dataset, split_train_val_dataset
from image.image_model import get_pretrained_vgg16, get_pretrained_resnet50, get_pretrained_efficientnet_b0
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler


def save_model_weights(model, model_name):
    path = f"{model_name}_weights.pth"
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


def load_pretrained_model(model_choice, num_classes=2, device="cpu"):
    path = f"{model_choice}_weights.pth"

    if model_choice == 'resnet50':
        model = get_pretrained_resnet50(num_classes=num_classes, freeze_features=False)
    elif model_choice == 'efficientnet':
        model = get_pretrained_efficientnet_b0(num_classes=num_classes, freeze_features=False)
    else:
        model = get_pretrained_vgg16(num_classes=num_classes, freeze_features=False)

    model.load_state_dict(torch.load(path, map_location=device))

    return model.to(device)


def train(train_image_path, train_class_path, device, num_epochs=10, optimizer_choice='adam', model_choice='vgg16'):
    train_images = preprocess_image(train_image_path)
    train_classes = get_classes(train_class_path)

    dataset = create_dataset(train_images, train_classes)
    train_loader, val_loader = split_train_val_dataset(dataset)

    if model_choice == 'efficientnet':
        model = get_pretrained_efficientnet_b0().to(device)
    elif model_choice == 'resnet50':
        model = get_pretrained_resnet50().to(device)
    else:
        model = get_pretrained_vgg16().to(device)

    if optimizer_choice == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)
    criterion = nn.CrossEntropyLoss()

    step_interval = 10

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        step_count = 0
        correct_predictions = 0
        total_predictions = 0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step_count += 1

            _, predicted = outputs.max(dim=1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            if (step + 1) % step_interval == 0:
                batch_accuracy = correct_predictions / total_predictions
                print(
                    f"Epoch[{epoch + 1}], Step[{step + 1}], Loss: {loss.item():.4f},"
                    f"Accuracy: {batch_accuracy * 100:.2f}"
                )
                correct_predictions = 0
                total_predictions = 0

        scheduler.step()

        avg_loss = total_loss / step_count
        print(f'Epoch[{epoch + 1}] completed. Average Loss: {avg_loss:.4f}')

    save_model_weights(model, model_choice)


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


def test(test_image_path, dataframe, model_choice, device):
    test_image = get_image(test_image_path)

    model = load_pretrained_model(model_choice)

    model.eval()
    with torch.no_grad():
        for user_id, image in test_image.items():
            image = image.to(device)

            image = image.unsqueeze(0)

            output = model(image)
            output = F.sigmoid(output)
            pred_class = torch.argmax(output, dim=1).item()  # to get the class with the highest value
            # You don't format anything here, just set it as int,
            # That way we know that were are all getting the same kind of output
            # ie, the genders have to be lower case on the vm
            # result = "Female" if pred_class == 1 else "Male"
            dataframe.loc[dataframe['userid'] == user_id, 'gender'] = pred_class

    return dataframe
