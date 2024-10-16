from image.image_preprocessing import preprocess_image
from image.image_utils import get_classes, get_image, create_dataloader
from image.image_model import VGG16
import torch.optim as optim


def test(train_image_path, test_image_path, train_class_path, test_class_path, device):
    train_images = preprocess_image(train_image_path)
    train_classes = get_classes(train_class_path)

    test_images = get_image(test_image_path)
    test_classes = get_classes(test_class_path)

    train_loader = create_dataloader(train_images, train_classes)
    test_loader = create_dataloader(test_image_path, test_classes)

    # This is for gender.
    model = VGG16(2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    losses = []
    accuracies = []
    step_interval = 10

    model.train()
    for epoch in range(100):
        total_loss = 0.0
        step_count = 0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backword()
            optimizer.step()

            total_loss += loss.item()
            step_count += 1

            if (step + 1) % step_interval == 0:
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == labels).sum().item() / labels.size(0)
                print(f'Epoch [{epoch + 1}], Step [{step + 1}], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}')

        avg_loss = total_loss / step_count
        print(f'Epoch [{epoch + 1}] completed. Average Loss: {avg_loss:.4f}')
