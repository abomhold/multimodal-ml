INPUT_PATH: str
OUTPUT_PATH: str
TRAIN_PATH = "training"
TEXT_DIR = "text"
IMAGE_DIR = "image"
LIKE_DIR = "relation"
LIWC_DIR = "LIWC"
PROFILE_DIR = "profile"

IMAGE_TRAIN_PATH = f"{INPUT_PATH}/training/image"
CLASS_TRAIN_PATH = f"{INPUT_PATH}/training/profile/profile.csv"


def get_configs():
    print(f"input path: {INPUT_PATH}")
    print(f"output path: {OUTPUT_PATH}")


if __name__ == '__main__':
    get_configs()
