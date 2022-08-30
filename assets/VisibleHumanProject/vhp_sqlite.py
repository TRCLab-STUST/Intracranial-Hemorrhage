import os
import sqlite3

# Path
ROOT_DIR = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
ASSETS_VHP_DIR = os.path.join(ASSETS_DIR, "VisibleHumanProject")

print(ROOT_DIR)

# Database
DB_VHP = os.path.join(ASSETS_VHP_DIR, "VHP_Label.db")

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


def create_connection(db_file):
    connect = None
    try:
        connect = sqlite3.connect(db_file)
        return connect
    except ConnectionError as e:
        print(e)

    return connect


def create_table(connection, create_table_sql):
    try:
        cursor = connection.cursor()
        cursor.execute(create_table_sql)
    except ConnectionError as e:
        print(e)


def insert(connection, table, data, columns=None):
    if columns is not None:
        columns = f"({', '.join(columns)})"

    data = ", ".join([f"'{col}'" if type(col) == str else f"{col}" for col in data])
    cursor = connection.cursor()
    cursor.execute(
        f"INSERT INTO {table} {columns} VALUES ({data})"
    )
    connection.commit()


def main():
    db_vhp = create_connection(DB_VHP)
    if db_vhp is not None:
        create_table(db_vhp, TABLE_VHP_ORGAN)
        create_table(db_vhp, TABLE_VHP_COLOR)
        create_table(db_vhp, TABLE_VHP_RANGE)

    vhp_label = os.path.join(ASSETS_VHP_DIR, "original", "color_with_mandarin.txt")
    with open(vhp_label, "r") as label_file:
        label_data = label_file.read()
        for idx, line in enumerate(label_data.splitlines()[1:-1], 1):
            columns = line.split("\t")
            print(columns)
            insert(db_vhp, "organ",
                   data=[columns[0]] + columns[6:9],
                   columns=["name", "roughsystem", "subsystem", "mandarin"])
            insert(db_vhp, "color",
                   data=[idx] + [int(n) for n in columns[1:4]],
                   columns=["organ_id", "red", "green", "blue"])
            insert(db_vhp, "range",
                   data=[idx] + [int(n) for n in columns[4:6]],
                   columns=["organ_id", "begin", "end"])

    db_vhp.close()


if __name__ == '__main__':
    main()
