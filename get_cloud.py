import os

import gdown

import config


def download_cloud_assets():
    if not os.path.exists("cloud_assets"):
        os.mkdir("cloud_assets")
    gdown.download(config.CLOUD_ASSETS_URL + config.CLOUD_ASSETS_ID, output="cloud_assets.zip", quiet=False, fuzzy=True)
    os.system("unzip cloud_assets.zip -d cloud_assets")


if __name__ == '__main__':
    download_cloud_assets()
