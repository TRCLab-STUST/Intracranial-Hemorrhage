import sqlite3


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