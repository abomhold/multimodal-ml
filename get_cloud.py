import gdown

CLOUD_ASSETS_URL = "https://drive.google.com/uc?id="
CLOUD_ASSETS_ID = "1DQkXlgCTZc0ILO-pTBjPjkiZWZdpZaKQ"
gdown.download(CLOUD_ASSETS_URL + CLOUD_ASSETS_ID, output="cloud_assets.zip", quiet=False, fuzzy=True)
