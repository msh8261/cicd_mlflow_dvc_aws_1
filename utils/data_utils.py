"""module for prepare dataframe of data."""
import os
import string
from pathlib import Path

import pandas as pd


def get_image_df(imags_root: string) -> pd.DataFrame:
    """get images path and return dataframe of them."""
    root_normal = os.path.join(imags_root, "NORMAL")
    # make dataframe of normal images
    img_df_normal = pd.DataFrame(
        {"img_path": list(map(str, Path(root_normal).rglob("*.jpeg")))},
    )
    # labels with 0 value for normal label
    img_df_normal["label"] = img_df_normal["img_path"].map(lambda p: int(0))
    root_pneumonia = os.path.join(imags_root, "PNEUMONIA")
    # make dataframe of pneumonia images
    img_df_pneumonia = pd.DataFrame(
        {"img_path": list(map(str, Path(root_pneumonia).rglob("*.jpeg")))},
    )
    # labels with 1 for virus label
    img_df_pneumonia["label"] = img_df_pneumonia["img_path"].map(
        lambda fp: int("virus" in os.path.basename(fp).lower())
    )
    # labels with 2 for bactria label
    img_df_pneumonia["label"].replace(0, int(2), inplace=True)
    # concatenate two dataframes
    img_df = pd.concat([img_df_normal, img_df_pneumonia], axis=0)
    # shuffle the DataFrame rows
    img_df = img_df.sample(frac=1)
    return img_df


# get_image_df("./data/raw/Chest_X_Ray/train")
