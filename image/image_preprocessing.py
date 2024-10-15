from PIL import Image
import os
from torchvision.transforms import v2
from image_config import WIDTH, HEIGHT

transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_image(image_dir):
    images = []
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)

        try:
            image = Image.open(image_path)
            resized_image = resize_image(image)
            augmented_image = transforms(resized_image)
            images.append(augmented_image)
        except IOError:
            print(f"Error opening image: {image_path}")

    return images


def resize_image(image):
    img_width, img_height = image.size
    if img_width != WIDTH or img_height != HEIGHT:
        resized_image = image.resize((WIDTH, HEIGHT), Image.LANCZOS)
    else:
        resized_image = image
    return resized_image
