import sys
from pathlib import Path
import config
import text.preprocessing
import preprocessing as pre
import pandas as pd
import postprocessing as post

import image.image_testrun as image_testrun
import torch

import image.image_testrun as image_testrun
import torch

device = "gpu" if torch.cuda.is_available() else "cpu"


def main():
    if len(sys.argv) == 2:
        config.INPUT_PATH = sys.argv[1]
        config.OUTPUT_PATH = sys.argv[2]

    input_path = Path(config.INPUT_PATH)
    data = pre.main(input_path.joinpath(config.PROFILE_PATH))

    data = image_testrun.test(input_path.joinpath(config.IMAGE_DIR), data, device)
    #    result.to_csv('result.csv', index=False)
    post.write_xml(Path(config.OUTPUT_PATH), data)


if __name__ == "__main__":
    main()
