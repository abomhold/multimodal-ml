import torch
import pandas as pd
import image.image_testrun as image_testrun
import config
import like.predict
import preprocessing as pre
import postprocessing as post
from pathlib import Path
import argparse
import text.main as text
from text import personality_prediction

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'resnet50'


def parse_args():
    parser = argparse.ArgumentParser(description='Process input and output paths')
    parser.add_argument('-i', '--input', dest='input_path', default='input',
                        help='Input path (default: input)')
    parser.add_argument('-o', '--output', dest='output_path', default='output',
                        help='Output path (default: output)')

    args = parser.parse_args()
    return args.input_path, args.output_path


def main():
    print("Starting...")

    input_dir, output_dir = parse_args()
    config.set_configs(input_dir, output_dir)
    config.get_configs()

    data = pre.main()
    print("Starting Image Prediction...")
    image_df = image_testrun.test(config.IMAGE_DIR, data.copy(), model_name, device)
    print("Starting Text Prediction...")
    personality_df = text.main(Path(config.TEXT_DIR), data.copy())
    print("Starting Like Prediction...")
    like_df = like.predict.predict_all(relation_path=config.LIKE_PATH, data=data.copy())
    print("Combining Data...")

    combined_df = pd.merge(
        image_df.loc[:, ['userid', 'gender']],
        personality_df.loc[:, ['userid', 'ope', 'con', 'ext', 'agr', 'neu']],
        on='userid'
    )
    combined_df = pd.merge(
        combined_df,
        like_df.loc[:, ['userid', 'age']],
        on='userid'
    )

    post.write_xml(Path(config.OUTPUT_PATH), combined_df)
    print("Done!")


if __name__ == "__main__":
    main()
