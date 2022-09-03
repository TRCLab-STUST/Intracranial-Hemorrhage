import os
import PATH
import sqlite as sql
from tqdm import tqdm

# Database
DB_VHP = os.path.join(PATH.ASSETS_VHP_DIR, "VHP_Label.db")

# SQL Table
TABLE_VHP_ORGAN = """ 
    CREATE TABLE IF NOT EXISTS organ (
        id integer PRIMARY KEY autoincrement,
        name text NOT NULL,
        roughsystem text NOT NULL,
        subsystem text NOT NULL,
        mandarin text NOT NULL
    ); """

TABLE_VHP_COLOR = """ 
    create table if not exists color (
        organ_id integer not null
            constraint color_pk
                primary key
            constraint color_organ_null_fk
                references organ (id),
        red      integer not null,
        green    integer not null,
        blue     integer not null
    ); """

TABLE_VHP_RANGE = """ 
    create table if not exists range (
        organ_id integer not null
            constraint range_pk
                primary key
            constraint range_organ_null_fk
                references organ (id),
        begin   integer not null,
        end     integer not null
    ); """


def main():
    db_vhp = sql.create_connection(DB_VHP)
    if db_vhp is not None:
        sql.create_table(db_vhp, TABLE_VHP_ORGAN)
        sql.create_table(db_vhp, TABLE_VHP_COLOR)
        sql.create_table(db_vhp, TABLE_VHP_RANGE)

    vhp_label = os.path.join(PATH.DATASET_VHP_ORIGINAL, "color_with_mandarin.txt")
    with open(vhp_label, "r") as label_file:
        label_data = label_file.read()
        for idx, line in tqdm(enumerate(label_data.splitlines()[1:-1], 1)):
            columns = line.split("\t")
            sql.insert(db_vhp, "organ",
                       data=[columns[0]] + columns[6:9],
                       columns=["name", "roughsystem", "subsystem", "mandarin"])
            sql.insert(db_vhp, "color",
                       data=[idx] + [int(n) for n in columns[1:4]],
                       columns=["organ_id", "red", "green", "blue"])
            sql.insert(db_vhp, "range",
                       data=[idx] + [int(n) for n in columns[4:6]],
                       columns=["organ_id", "begin", "end"])

    db_vhp.close()


if __name__ == '__main__':
    main()
