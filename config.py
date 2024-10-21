INPUT_PATH: str
OUTPUT_PATH: str
TRAIN_PATH = "training"
TEXT_DIR = "text"
IMAGE_DIR = "image"
LIKE_DIR = "relation"
LIWC_DIR = "LIWC"
PROFILE_DIR = "profile"

IMAGE_TRAIN_PATH = "dataset/training/image"
CLASS_TRAIN_PATH = "dataset/training/profile/profile.csv"
IMAGE_TEST_PATH = "dataset/public-test-data/image"
CLASS_TEST_PATH = "dataset/public-test-data/profile/profile.csv"


def get_configs():
    print(f"input path: {INPUT_PATH}")
    print(f"output path: {OUTPUT_PATH}")


if __name__ == '__main__':
    get_configs()
