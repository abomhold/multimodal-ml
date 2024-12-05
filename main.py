import sys
import torch
import image.image_testrun as image_testrun
import config
import like.predict
import preprocessing as pre
import postprocessing as post
import text.main as text
import os
from pathlib import Path
import argparse
import get_cloud
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'resnet50'


def parse_args():
    parser = argparse.ArgumentParser(description='Process input and output paths')
    parser.add_argument('-i', '--input', dest='input_path', default='input',
                       help='Input path (default: input)')
    parser.add_argument('-o', '--output', dest='output_path', default='output',
                       help='Output path (default: output)')

    args = parser.parse_args()

    config.INPUT_PATH = args.input_path
    config.OUTPUT_PATH = args.output_path

def main():
    print("Starting...")

    
    parse_args()
    config.set_configs()
    config.get_configs()

    get_cloud()    

    data = pre.main()
    # text_df = text.main(config.TEXT_DIR, data.copy())
    # image_df = image_testrun.test(config.IMAGE_DIR, data.copy(), model_name, device)
    like_df = like.predict.predict_all(relation_path=config.LIKE_PATH, data=data.copy())
    # combined_df = post.majority(text_df, image_df, like_df)
    post.write_xml(config.OUTPUT_PATH, like_df)
    print("Done!")


if __name__ == "__main__":
    main()
