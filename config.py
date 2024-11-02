from pathlib import Path

INPUT_PATH: Path = Path("/mnt/input")
OUTPUT_PATH: Path = Path("/mnt/output")

TEST_PATH = Path("public-test-data")

TEXT_DIR = INPUT_PATH.joinpath("text")
IMAGE_DIR = INPUT_PATH.joinpath("image")
LIKE_DIR = INPUT_PATH.joinpath("relation")
LIWC_PATH = INPUT_PATH.joinpath("LIWC/LIWC.csv")
PROFILE_PATH = INPUT_PATH.joinpath("profile/profile.csv")

CLOUD_ASSETS_URL = "https://drive.google.com/uc?id="
CLOUD_ASSETS_ID = "1DQkXlgCTZc0ILO-pTBjPjkiZWZdpZaKQ"


def get_configs():
    print(f"input path: {INPUT_PATH}")
    print(f"output path: {OUTPUT_PATH}\n")
    print(f"text dir: {TEXT_DIR}")
    print(f"image dir: {IMAGE_DIR}")
    print(f"like dir: {LIKE_DIR}")
    print(f"LIWC path: {LIWC_PATH}")
    print(f"profile path: {PROFILE_PATH}\n")
    print(f"[LOCAL]\ntest path: {TEST_PATH}")


if __name__ == '__main__':
    # set_paths("input", "output")
    get_configs()

# def set_paths(the_input: str, the_output: str):
#     global INPUT_PATH, OUTPUT_PATH, TEXT_DIR, IMAGE_DIR, LIKE_DIR, LIWC_PATH, PROFILE_PATH, TEST_PATH
#
#     try:
#         INPUT_PATH = Path(the_input)
#         OUTPUT_PATH = Path(the_output)
#     except Exception as e:
#         INPUT_PATH = Path("input")
#         OUTPUT_PATH = Path("output")
#         print(f"PATH ERROR: {e}")
#
#     TEXT_DIR = INPUT_PATH.joinpath("text")
#     IMAGE_DIR = INPUT_PATH.joinpath("image")
#     LIKE_DIR = INPUT_PATH.joinpath("like")
#     LIWC_PATH = INPUT_PATH.joinpath("LIWC/LIWC.csv")
#     PROFILE_PATH = INPUT_PATH.joinpath("profile/profile.csv")
