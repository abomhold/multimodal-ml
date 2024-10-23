import sys
import config

from text import text_main

import pre_processing as pre
import image.image_testrun as image_testrun
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# def collate_data():


def main():
    config.INPUT_PATH = sys.argv[1]
    config.OUTPUT_PATH = sys.argv[2]

    

    pre.build_baseline()

    text_main.main()
    image_testrun.test(config.IMAGE_TEST_PATH, config.CLASS_TEST_PATH, device)


if __name__ == "__main__":
    main()
