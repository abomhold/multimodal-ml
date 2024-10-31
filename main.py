import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
############################################################################################################
# DON'T CHANGE ANYTHING ABOVE THIS
############################################################################################################

import torch
import config
import postprocessing as post
import preprocessing as pre
import text.main as text

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'resnet50'


# LEAVE THIS UNTIL AFTER 0900, I need to be graded
def main():
    print("Starting...")
    data = pre.main()
    config.get_configs()
    text_df = text.main(config.TEXT_DIR, data.copy())
    # image_df = image_testrun.test(config.IMAGE_DIR, data.copy(), model_name, device)
    # like_df = data
    # combined_df = post.majority(text_df, image_df, like_df)
    post.write_xml(config.OUTPUT_PATH, text_df)
    print("Done!")


if __name__ == "__main__":
    main()
