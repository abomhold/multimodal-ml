INPUT_PATH = "/mnt/input"
OUTPUT_PATH = "/mnt/output"
TEXT_DIR = "/mnt/input/text"
IMAGE_DIR = "/mnt/input/image"
LIKE_PATH = "/mnt/input/relation/relation.csv"
LIWC_PATH = "/mnt/input/LIWC/LIWC.csv"
PROFILE_PATH = "/mnt/input/profile/profile.csv"
CLOUD_DIR = "/mnt/input/cloud_assets"
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
