import sys
from pathlib import Path
import pandas as pd
import config
import text
import image
import like
import preprocessing as pre
import postprocessing as post

import image.image_testrun as image_testrun
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    if len(sys.argv) == 2:
        config.INPUT_PATH = sys.argv[1]
        config.OUTPUT_PATH = sys.argv[2]

    input_path = Path(config.INPUT_PATH)
    # todo: add LWIC to preprocessing step
    data = pre.main(input_path.joinpath(config.PROFILE_PATH))

    # TEXT
    text_df = text.preprocessing.main(input_path.joinpath(config.TEXT_DIR), data.copy())
    text_df = text.test.main(input_path.joinpath(config.TEXT_DIR), text_df)
    print(text_df.head())

    # IMAGES
    image_df = image.preprocessing.main(input_path.joinpath(config.TEXT_DIR), data.copy())
    image_df = image.test.main(input_path.joinpath(config.TEXT_DIR), image_df)
    print(image_df.head())

    # LIKES
    like_df = like.preprocessing.main(input_path.joinpath(config.TEXT_DIR), data.copy())
    like_df = like.test.main(input_path.joinpath(config.TEXT_DIR), like_df)
    print(like_df.head())

    combined_df = post.majority(text_df, image_df, like_df)
    post.write_xml(Path(config.OUTPUT_PATH), data)


if __name__ == "__main__":
    main()
