import os

from tqdm import tqdm

import PATH
from assets import sqlite as sql

TABLE_VHP_TYPE = """
    CREATE TABLE IF NOT EXISTS type
    (
        id   INTEGER NOT NULL
            CONSTRAINT type_pk
                PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL
    );
"""

TABLE_VHP_IMAGE = """
    CREATE TABLE IF NOT EXISTS original
    (
        id      INTEGER NOT NULL
            CONSTRAINT original_pk
                PRIMARY KEY AUTOINCREMENT,
        type_id INTEGER NOT NULL
            CONSTRAINT original_type_null_fk
                REFERENCES type (id),
        folder  TEXT    NOT NULL,
        name    TEXT    NOT NULL
    );
"""

IMAGE_TYPE_LIST = ["CT", "MR", "SEGMENTED", "CRYO"]
ORIGINAL_DIR_NAME_LIST = [
    "(VKH) CT Images (494 X 281)",
    "(VKH) MR Images (494 X 281)",
    "(VKH) Segmented Images (1,000 X 570)",
    "(VKH) Anatomical Images (1,000 X 570)"
]


def main():
    db_vhp = sql.create_connection("VHP.db")
    sql.create_table(db_vhp, TABLE_VHP_IMAGE)
    sql.create_table(db_vhp, TABLE_VHP_TYPE)

    for image_type in IMAGE_TYPE_LIST:
        sql.insert(db_vhp, "type",
                   data=[image_type],
                   columns=["type"])

    for idx, folder_name in enumerate(ORIGINAL_DIR_NAME_LIST, start=1):
        target_path = os.path.join(PATH.DATASET_VHP_ORIGINAL, folder_name)
        image_name_list = os.listdir(target_path)
        for image in tqdm(image_name_list):
            sql.insert(db_vhp, "original",
                       data=[idx, ORIGINAL_DIR_NAME_LIST[idx - 1], image],
                       columns=["type_id", "folder", "name"])


if __name__ == '__main__':
    main()
