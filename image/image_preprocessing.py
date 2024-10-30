from PIL import Image
import torch
from torchvision.transforms import v2
from image.image_config import WIDTH, HEIGHT

transforms = v2.Compose([
    v2.Resize([WIDTH, HEIGHT]),
    v2.RandomHorizontalFlip(p=0.3),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_image(image_dir):
    image_dict = {}
    for image_file in image_dir.iterdir():
        try:
            original_image = Image.open(image_file)
            image_copy = original_image.copy()
            augmented_image = transforms(image_copy)
            image_name = image_file.stem
            image_dict[image_name] = augmented_image
        except IOError:
            print(f"Error opening image: {image_file}")

    return image_dict
