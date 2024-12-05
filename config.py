c:
OUTPUT_PATH: str 
INPUT_PATH: str
TEXT_DIR = f"{INPUT_PATH}/text"
IMAGE_DIR = f"{INPUT_PATH}/image"
LIKE_PATH = f"{INPUT_PATH}/relation/relation.csv"
LIWC_PATH = f"{INPUT_PATH}/LIWC/LIWC.csv"
PROFILE_PATH = f"{INPUT_PATH}/profile/profile.csv"


def get_configs():
    print(f"input path: {INPUT_PATH}")
    print(f"output path: {OUTPUT_PATH}\n")
    print(f"text dir: {TEXT_DIR}")
    print(f"image dir: {IMAGE_DIR}")
    print(f"like dir: {LIKE_PATH}")
    print(f"LIWC path: {LIWC_PATH}")
    print(f"profile path: {PROFILE_PATH}\n")


if __name__ == '__main__':
    get_configs()
