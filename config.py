
PROFILE_PATH = "tcss555/training/profile/profile.csv"
INPUT_PATH: str
OUTPUT_PATH: str
IMAGE_TRAIN_PATH = "dataset/training/image"
CLASS_TRAIN_PATH = "dataset/training/profile/profile.csv"
IMAGE_TEST_PATH = "dataset/public-test-data/image"
CLASS_TEST_PATH = "dataset/public-test-data/profile/profile.csv"


def get_configs():
    print(f"input path: {INPUT_PATH}")
    print(f"output path: {OUTPUT_PATH}")


if __name__ == '__main__':
    get_configs()
