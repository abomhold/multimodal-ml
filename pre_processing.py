import pandas as pd
from config import PROFILE_PATH


def get_most_gender():
    data = pd.read_csv(PROFILE_PATH)
    mode_gender = data["gender"].mode()[0]
    return mode_gender


def get_most_age():
    data = pd.read_csv(PROFILE_PATH)
    mode_age = data["age"].mode()[0]
    return mode_age


def get_avg_personality():
    data = pd.read_csv(PROFILE_PATH)
    avg_ope = data["ope"].mean()
    ave_con = data["con"].mean()
    ave_ext = data["ext"].mean()
    ave_agr = data["agr"].mean()
    ave_neu = data["neu"].mean()

    return avg_ope, ave_con, ave_ext, ave_agr, ave_neu
