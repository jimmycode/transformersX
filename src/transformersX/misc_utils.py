from datetime import datetime


def append_timestamp_to_dir(dir_path):
    dir_path = dir_path.strip()
    if dir_path.endswith("/"):
        dir_path = dir_path[:-1]
    if dir_path == "":
        raise ValueError("Empty directory path.")

    timestamp_str = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    new_dir = dir_path + "_" + timestamp_str

    return new_dir
