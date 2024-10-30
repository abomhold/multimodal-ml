from pathlib import Path

INPUT_PATH: Path
OUTPUT_PATH: Path

TEST_PATH = Path("public-test-data")

TEXT_DIR: Path
IMAGE_DIR: Path
LIKE_DIR: Path
LIWC_DIR: Path
PROFILE_PATH: Path


# IMAGE_TRAIN_PATH = f"training/image"
# CLASS_TRAIN_PATH = f"training/profile/profile.csv"


def get_configs():
    print(f"input path: {INPUT_PATH}")
    print(f"output path: {OUTPUT_PATH}\n")
    print(f"text dir: {TEXT_DIR}")
    print(f"image dir: {IMAGE_DIR}")
    print(f"like dir: {LIKE_DIR}")
    print(f"LIWC dir: {LIWC_DIR}")
    print(f"profile path: {PROFILE_PATH}\n")
    print(f"[LOCAL]\ntest path: {TEST_PATH}")


def set_paths(the_input: str, the_output: str):
    global INPUT_PATH, OUTPUT_PATH, TEXT_DIR, IMAGE_DIR, LIKE_DIR, LIWC_DIR, PROFILE_PATH, TEST_PATH

    try:
        INPUT_PATH = Path(the_input)
        OUTPUT_PATH = Path(the_output)
    except Exception as e:
        INPUT_PATH = Path("input")
        OUTPUT_PATH = Path("output")
        print(f"PATH ERROR: {e}")

    TEXT_DIR = INPUT_PATH.joinpath("text")
    IMAGE_DIR = INPUT_PATH.joinpath("image")
    LIKE_DIR = INPUT_PATH.joinpath("like")
    LIWC_DIR = INPUT_PATH.joinpath("LIWC")
    PROFILE_PATH = INPUT_PATH.joinpath("profile/profile.csv")


if __name__ == '__main__':
    get_configs()
