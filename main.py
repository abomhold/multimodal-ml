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
model_name = 'resnet50'


def main():
    if len(sys.argv) == 2:
        config.INPUT_PATH = sys.argv[1]
        config.OUTPUT_PATH = sys.argv[2]

    input_path = Path(config.INPUT_PATH)
    # todo: add LWIC to preprocessing step
    #data = pre.main(input_path.joinpath(config.PROFILE_PATH))
    data = pre.main(Path(config.TEST_PATH).joinpath(config.PROFILE_PATH))

    # TEXT
    #text_df = text.preprocessing.main(input_path.joinpath(config.TEXT_DIR), data.copy())
    #text_df = text.test.main(input_path.joinpath(config.TEXT_DIR), text_df)
    #print(text_df.head())

    # IMAGES
#    image_testrun.train(
#        input_path.joinpath(config.IMAGE_DIR),
#        input_path.joinpath(config.PROFILE_PATH),
#        device,
#        num_epochs=10,
#        model_choice=model_name
#    )
    image_df = image_testrun.test(Path(config.TEST_PATH).joinpath(config.IMAGE_DIR), data.copy(), model_name, device)
    print(image_df.head())

    # LIKES
#    like_df = like.preprocessing.main(input_path.joinpath(config.TEXT_DIR), data.copy())
#    like_df = like.test.main(input_path.joinpath(config.TEXT_DIR), like_df)
#    print(like_df.head())

    #combined_df = post.majority(text_df, image_df, like_df)
    post.write_xml(Path(config.OUTPUT_PATH), image_df)


if __name__ == "__main__":
    main()
