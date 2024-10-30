INPUT_PATH: str = "training/"
OUTPUT_PATH: str = "output/"
TRAIN_PATH = "training"
TEXT_DIR = "text"
IMAGE_DIR = "image"
LIKE_DIR = "relation"
LIWC_DIR = "LIWC"
PROFILE_PATH = "profile/profile.csv"
TEST_PATH = "dataset/public-test-data/"

#IMAGE_TRAIN_PATH = f"training/image"
#CLASS_TRAIN_PATH = f"training/profile/profile.csv"


def get_configs():
    print(f"input path: {INPUT_PATH}")
    print(f"output path: {OUTPUT_PATH}")


if __name__ == '__main__':
    get_configs()
