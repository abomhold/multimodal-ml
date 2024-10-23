from PIL import Image
import os
import torch
from torchvision.transforms import v2
from image.image_config import WIDTH, HEIGHT

transforms = v2.Compose([
    v2.Resize([WIDTH, HEIGHT]),
    v2.RandomHorizontalFlip(p=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_image(image_dir):
    image_dict = {}
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)

        try:
            image = Image.open(image_path)
            augmented_image = transforms(image)
            image_name = image_file.removesuffix('.jpg')
            image_dict[image_name] = augmented_image
        except IOError:
            print(f"Error opening image: {image_path}")

    return image_dict
