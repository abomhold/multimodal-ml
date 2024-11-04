INPUT_PATH = "/tmp/input"
OUTPUT_PATH = "/tmp/output"
TEXT_DIR = "/tmp/input/text"
IMAGE_DIR = "/tmp/input/image"
LIKE_PATH = "/tmp/input/relation/relation.csv"
LIWC_PATH = "/tmp/input/LIWC/LIWC.csv"
PROFILE_PATH = "/tmp/input/profile/profile.csv"
CLOUD_DIR = "/tmp/input/cloud_assets"
PROJECT_ROOT = "/home"


def get_configs():
    print(f"input path: {INPUT_PATH}")
    print(f"output path: {OUTPUT_PATH}\n")
    print(f"text dir: {TEXT_DIR}")
    print(f"image dir: {IMAGE_DIR}")
    print(f"like dir: {LIKE_PATH}")
    print(f"LIWC path: {LIWC_PATH}")
    print(f"profile path: {PROFILE_PATH}\n")
    print(f"cloud assets: {CLOUD_DIR}")


if __name__ == '__main__':
    get_configs()
