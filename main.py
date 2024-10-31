import sys
from pathlib import Path
# import pandas as pd
import config
import text.main as text
import image
import like
import preprocessing as pre
import postprocessing as post

import image.image_testrun as image_testrun
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'resnet50'


def parse_args(args: list):
    if len(args) == 3:
        config.set_paths(args[1], args[2])
    else:
        config.set_paths(config.TEST_PATH, "output")
    # print(config.get_configs())


def main():
    parse_args(sys.argv)
    data = pre.main()
    text_df = text.main(config.TEXT_DIR, data.copy())
    # image_df = image_testrun.test(config.IMAGE_DIR, data.copy(), model_name, device)
    # like_df = like.test.main(config.LIKE_DIR, data.copy())
    # combined_df = post.majority(text_df, image_df, like_df)
    post.write_xml(config.OUTPUT_PATH, text_df)
    print("Done!")

if __name__ == "__main__":
    main()
