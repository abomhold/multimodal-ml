OUTPUT_PATH: str = "output"
INPUT_PATH: str = "input"
TEXT_DIR: str = ""
IMAGE_DIR: str = ""
LIKE_PATH: str = ""
LIWC_PATH: str = ""
PROFILE_PATH: str = ""

def get_configs():
    print(f"input path: {INPUT_PATH}")
    print(f"output path: {OUTPUT_PATH}")
    print(f"text dir: {TEXT_DIR}")
    print(f"image dir: {IMAGE_DIR}")
    print(f"like dir: {LIKE_PATH}")
    print(f"LIWC path: {LIWC_PATH}")
    print(f"profile path: {PROFILE_PATH}\n")


def set_configs():
    global TEXT_DIR, IMAGE_DIR, LIKE_PATH, LIWC_PATH, PROFILE_PATH, OUTPUT_PATH, INPUT_PATH
    TEXT_DIR = INPUT_PATH + "text"
    IMAGE_DIR = INPUT_PATH + "image"
    LIKE_PATH = INPUT_PATH + "relation/relation.csv"
    LIWC_PATH = INPUT_PATH + "LIWC/LIWC.csv"
    PROFILE_PATH = INPUT_PATH + "profile/profile.csv"

if __name__ == '__main__':
    get_configs()
