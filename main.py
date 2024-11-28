import torch
import image.image_testrun as image_testrun
import config
import like.predict
import preprocessing as pre
import postprocessing as post
import text.main as text
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'resnet50'


def main():
    print("Starting...")

    data = pre.main()
    # text_df = text.main(config.TEXT_DIR, data.copy())
    # image_df = image_testrun.test(config.IMAGE_DIR, data.copy(), model_name, device)
    like_df = like.predict.predict_all(relation_path=config.LIKE_PATH, data=data.copy())
    # combined_df = post.majority(text_df, image_df, like_df)
    post.write_xml(config.OUTPUT_PATH, like_df)
    print("Done!")


if __name__ == "__main__":
    main()
