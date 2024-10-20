
PROFILE_PATH = "tcss555/training/profile/profile.csv" 
INPUT_PATH: str
OUTPUT_PATH: str = "output"


def get_configs():
    print(f"input path: {INPUT_PATH}")
    print(f"output path: {OUTPUT_PATH}")


if __name__ == '__main__':
    get_configs()
