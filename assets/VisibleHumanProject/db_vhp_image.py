import os

from tqdm import tqdm

import PATH
import sqlite as sql

TABLE_VHP_TYPE = """
    create table if not exists type
    (
        id   integer not null
            constraint type_pk
                primary key autoincrement,
        type integer not null
    );
"""

TABLE_VHP_IMAGE = """
    create table if not exists original
    (
        id      integer not null
            constraint original_pk
                primary key autoincrement,
        type_id integer not null
            constraint original_type_null_fk
                references type (id),
        folder  TEXT    not null,
        name    TEXT    not null
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
