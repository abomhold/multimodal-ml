import sys
import config
import pre_processing as pre
import image.image_testrun as image_testrun
import torch


device = "gpu" if torch.cuda.is_available() else "cpu"


def main():
    config.INPUT_PATH = sys.argv[1]
    config.OUTPUT_PATH = sys.argv[2]
    pre.build_baseline()

    # Running the test for image classification for gender
    image_testrun.test(config.INPUT_PATH, device)


if __name__ == "__main__":
    main()
