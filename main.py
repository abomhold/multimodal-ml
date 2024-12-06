import torch
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

    input, output = parse_args()
    config.set_configs(input, output)
    config.get_configs()

    data = pre.main()
    print("Starting Image Prediction...")
    image_df = image_testrun.test(config.IMAGE_DIR, data.copy(), model_name, device)
    personality_df = text.main(Path(config.TEXT_DIR), data.copy())
    like_df = like.predict.predict_all(relation_path=config.LIKE_PATH, data=data.copy())
    combined_df = (personality_df['arg','con','ext','agr','neu']
                   .merge(image_df['gender'], on='userid', how='inner')
                   .merge(like_df['age'], on='id', how='inner'))

    post.write_xml(Path(config.OUTPUT_PATH), like_df)
    print("Done!")
#./tcss555 -i /data/public-test-data/ -o ~/slavam/

if __name__ == "__main__":
    main()
