import glob2
import os.path
from assets import sqlite as sql
from tqdm import tqdm
import PATH

ich_dataset_type_1 = ["non-progression", "progression"]
ich_dataset_type_2 = ["probably-non-progression", "probably-progression"]
ich_dataset_type_3 = ["validation-dicom", "validation-mask"]

V2_DATASET_EXCEPTION = {
    "Case 246-1271.tif": "Case 246-127-1.tif"
}

ICH_TYPE_LIST = ich_dataset_type_1 + ich_dataset_type_2 + ich_dataset_type_3
ICH_STAGE_LIST = ["Baseline", "Follow-UP"]
FILE_TYPE_LIST = ["dcm", "tif"]

TABLE_ICH_TYPE = """
    CREATE TABLE IF NOT EXISTS dataset_type
    (
        id   INTEGER NOT NULL
            CONSTRAINT dataset_type_pk
                PRIMARY KEY AUTOINCREMENT,
        type INTEGER NOT NULL
    );
"""

TABLE_FILE_TYPE = """
    create table if not exists file_type
    (
        id   INTEGER
            constraint file_type_pk
                primary key autoincrement,
        type TEXT not null
    );
"""

TABLE_ICH_STAGE = """
    create table if not exists stage
    (
        id    INTEGER
            constraint stage_pk
                primary key autoincrement,
        stage TEXT not null
    );
"""

TABLE_ICH_DICOM_IMAGE = """
    create table if not exists  dicom_image
    (
        id        INTEGER
            constraint dicom_image_pk
                primary key autoincrement,
        dataset_type_id   INTEGER not null
            constraint dicom_image_dataset_type_id_fk
                references dataset_type,
        folder    TEXT    not null,
        name      TEXT    not null,
        file_type_id INTEGER    not null
            constraint dicom_image_file_type_id_fk
                references file_type
    );
"""

TABLE_ICH_DICOM_INFO = """
    create table if not exists dicom_info
    (
        id       INTEGER
            constraint dicom_info_pk
                primary key autoincrement,
        image_id INTEGER not null
            constraint dicom_info_dicom_image_id_fk
                references dicom_image,
        stage_id INTEGER not null
            constraint dicom_info_stage_id_fk
                references stage,
        ich_case   INTEGER not null,
        image_index  INTEGER not null
    );
"""

TABLE_ICH_MASK_IMAGE = """
    create table if not exists  mask_image
    (
        id        INTEGER
            constraint mask_image_pk
                primary key autoincrement,
        dataset_type_id   INTEGER not null
            constraint mask_image_dataset_type_id_fk
                references dataset_type,
        folder    TEXT    not null,
        name      TEXT    not null,
        file_type_id INTEGER    not null
            constraint mask_image_file_type_id_fk
                references file_type
    );
"""

TABLE_ICH_MASK_INFO = """
    create table if not exists mask_info
    (
        id       INTEGER
            constraint mask_info_pk
                primary key autoincrement,
        image_id INTEGER not null
            constraint mask_info_dicom_image_id_fk
                references dicom_image,
        stage_id INTEGER not null
            constraint mask_info_stage_id_fk
                references stage,
        ich_case   INTEGER not null,
        image_index  INTEGER not null,
        range_index  INTEGER not null,
        ivh INTEGER not null 
    );
"""


def check_dataset_exception(exception_map: dict, basename: str) -> bool:
    if basename in exception_map.keys():
        return True
    return False


def get_ich_dataset_stage(image_file_path):
    parent_dir = os.path.abspath(os.path.join(image_file_path, os.pardir))
    if "FU" in parent_dir or "Follow" in parent_dir or "-1" in parent_dir:
        return 2
    elif "BL" in parent_dir or "Baseline" in parent_dir or "-2" in parent_dir:
        return 1
    else:
        return 0


def setup_database():
    db = sql.create_connection("ICH.db")
    sql.create_table(db, TABLE_ICH_TYPE)
    sql.create_table(db, TABLE_FILE_TYPE)
    sql.create_table(db, TABLE_ICH_STAGE)
    sql.create_table(db, TABLE_ICH_DICOM_IMAGE)
    sql.create_table(db, TABLE_ICH_DICOM_INFO)
    sql.create_table(db, TABLE_ICH_MASK_IMAGE)
    sql.create_table(db, TABLE_ICH_MASK_INFO)
    for ich_type in ICH_TYPE_LIST:
        sql.insert(db, "dataset_type",
                   data=[ich_type],
                   columns=["type"])

    for file_type in FILE_TYPE_LIST:
        sql.insert(db, "file_type",
                   data=[file_type],
                   columns=["type"])

    for ich_stage in ICH_STAGE_LIST:
        sql.insert(db, "stage",
                   data=[ich_stage],
                   columns=["stage"])

    return db


def main():
    db_ich = setup_database()

    image_file_list = glob2.glob(os.path.join(PATH.DATASET_ICH_V2, "**/*.*"))
    dicom_image_id = 1
    mask_image_id = 1
    print("Create Image Tabel: ")
    for image_file_path in tqdm(image_file_list):
        basename = os.path.basename(image_file_path)
        parent_dir = os.path.abspath(os.path.join(image_file_path, os.pardir))

        if check_dataset_exception(V2_DATASET_EXCEPTION, basename):
            basename = new_filename = V2_DATASET_EXCEPTION[basename]
            new_filepath = os.path.abspath(os.path.join(image_file_path, os.pardir, new_filename))
            os.rename(image_file_path, new_filepath)
            image_file_path = new_filepath

        stage_id = get_ich_dataset_stage(image_file_path)
        filename, filetype = basename.split(" ")[-1].split(".")

        if "probably-non-progression" in image_file_path:
            dataset_type_id = ICH_TYPE_LIST.index("probably-non-progression") + 1
            case, case_image_index = filename.split("-")
            folder = parent_dir.replace(PATH.DATASET_ICH_V2, "")[1:]
            file_type_id = FILE_TYPE_LIST.index(filetype) + 1

            sql.insert(db_ich, "dicom_image",
                       data=[dataset_type_id, folder, basename, file_type_id],
                       columns=["dataset_type_id", "folder", "name", "file_type_id"])
            sql.insert(db_ich, "dicom_info",
                       data=[dicom_image_id, stage_id, int(case), int(case_image_index)],
                       columns=["image_id", "stage_id", "ich_case", "image_index"])
            dicom_image_id += 1
            pass

        elif "probably-progression" in image_file_path:
            dataset_type_id = ICH_TYPE_LIST.index("probably-progression") + 1
            case, case_image_index = filename.split("-")
            folder = parent_dir.replace(PATH.DATASET_ICH_V2, "")[1:]
            file_type_id = FILE_TYPE_LIST.index(filetype) + 1
            sql.insert(db_ich, "dicom_image",
                       data=[dataset_type_id, folder, basename, file_type_id],
                       columns=["dataset_type_id", "folder", "name", "file_type_id"])
            sql.insert(db_ich, "dicom_info",
                       data=[dicom_image_id, stage_id, int(case), int(case_image_index)],
                       columns=["image_id", "stage_id", "ich_case", "image_index"])
            dicom_image_id += 1
            pass

        elif "non-progression" in image_file_path:
            dataset_type_id = ICH_TYPE_LIST.index("non-progression") + 1
            folder = parent_dir.replace(PATH.DATASET_ICH_V2, "")[1:]
            case, case_image_id, range_with_type = filename.split("-")
            range_str = ''.join(filter(str.isdigit, range_with_type))
            range_str = "1" if range_str == "" else range_str
            mask_range = int(range_str)
            mask_ivh = 1 if "V" in range_with_type else 0
            file_type_id = FILE_TYPE_LIST.index(filetype) + 1
            sql.insert(db_ich, "mask_image",
                       data=[dataset_type_id, folder, basename, file_type_id],
                       columns=["dataset_type_id", "folder", "name", "file_type_id"])
            sql.insert(db_ich, "mask_info",
                       data=[int(mask_image_id), int(stage_id), int(case), int(case_image_id), int(mask_range), int(mask_ivh)],
                       columns=["image_id", "stage_id", "ich_case", "image_index", "range_index", "ivh"])

            mask_image_id += 1

        elif "progression" in image_file_path:
            dataset_type_id = ICH_TYPE_LIST.index("progression") + 1
            folder = parent_dir.replace(PATH.DATASET_ICH_V2, "")[1:]
            case, case_image_id, range_with_type = filename.split("-")
            range_str = ''.join(filter(str.isdigit, range_with_type))
            range_str = "1" if range_str == "" else range_str
            mask_range = int(range_str)
            mask_ivh = 1 if "V" in range_with_type else 0
            file_type_id = FILE_TYPE_LIST.index(filetype) + 1
            sql.insert(db_ich, "mask_image",
                       data=[dataset_type_id, folder, basename, file_type_id],
                       columns=["dataset_type_id", "folder", "name", "file_type_id"])
            sql.insert(db_ich, "mask_info",
                       data=[int(mask_image_id), int(stage_id), int(case), int(case_image_id), int(mask_range),
                             int(mask_ivh)],
                       columns=["image_id", "stage_id", "ich_case", "image_index", "range_index", "ivh"])

            mask_image_id += 1

        elif "validation-dicom" in image_file_path:
            dataset_type_id = ICH_TYPE_LIST.index("validation-dicom") + 1
            case, _, case_image_index = filename.replace("_", "-").split("-")
            folder = parent_dir.replace(PATH.DATASET_ICH_V2, "")[1:]
            file_type_id = FILE_TYPE_LIST.index(filetype) + 1

            sql.insert(db_ich, "dicom_image",
                       data=[dataset_type_id, folder, basename, file_type_id],
                       columns=["dataset_type_id", "folder", "name", "file_type_id"])
            sql.insert(db_ich, "dicom_info",
                       data=[dicom_image_id, stage_id, int(case), int(case_image_index)],
                       columns=["image_id", "stage_id", "ich_case", "image_index"])
            dicom_image_id += 1

        elif "validation-mask" in image_file_path:
            filename_split = filename.replace("_", "-").split("-")
            if len(filename_split) == 3:
                filename_split.append("1")

            dataset_type_id = ICH_TYPE_LIST.index("validation-mask") + 1
            folder = parent_dir.replace(PATH.DATASET_ICH_V2, "")[1:]
            case, _, case_image_id, range_with_type = filename_split
            range_str = ''.join(filter(str.isdigit, range_with_type))
            range_str = "1" if range_str == "" else range_str
            mask_range = int(range_str)
            mask_ivh = 1 if "V" in range_with_type else 0
            file_type_id = FILE_TYPE_LIST.index(filetype) + 1
            sql.insert(db_ich, "mask_image",
                       data=[dataset_type_id, folder, basename, file_type_id],
                       columns=["dataset_type_id", "folder", "name", "file_type_id"])
            sql.insert(db_ich, "mask_info",
                       data=[int(mask_image_id), int(stage_id), int(case), int(case_image_id), int(mask_range),
                             int(mask_ivh)],
                       columns=["image_id", "stage_id", "ich_case", "image_index", "range_index", "ivh"])

            mask_image_id += 1


if __name__ == '__main__':
    main()
