import torch

import config
import like.bayes
import preprocessing as pre
import postprocessing as post

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'resnet50'


def main():
    print("Starting...")
    data = pre.main()
    # text_df = text.main(config.TEXT_DIR, data.copy())
    # image_df = image_testrun.test(config.IMAGE_DIR, data.copy(), model_name, device)
    like_df = like.bayes.predict_gender(relation_path=config.LIKE_PATH, data=data.copy())
    # combined_df = post.majority(text_df, image_df, like_df)
    post.write_xml(config.OUTPUT_PATH, like_df)
    print("Done!")


if __name__ == "__main__":
    main()
