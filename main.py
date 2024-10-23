import sys
from pathlib import Path
import config
import text.preprocessing
import preprocessing as pre


# import image.image_testrun as image_testrun
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"


# def collate_data():


def main():
    if len(sys.argv) == 2:
        config.INPUT_PATH = sys.argv[1]
        config.OUTPUT_PATH = sys.argv[2]

    input_path = Path(config.INPUT_PATH)
    data = pre.main(input_path.joinpath(config.PROFILE_PATH))

    # Text preprocessing
    data = text.preprocessing.main(input_path.joinpath(config.TEXT_DIR), data)
    print(data)

    #
    # image_testrun.test(config.IMAGE_TEST_PATH, config.CLASS_TEST_PATH, device)


if __name__ == "__main__":
    main()
