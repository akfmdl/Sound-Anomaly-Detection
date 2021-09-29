import time, datetime, json
import numpy as np

def milli_time():
    return int(round(time.time() * 1000))

def get_current_date():
    now = datetime.datetime.now()
    return now.strftime('%y%m%d')

def dict_to_list(src: dict):
    result_ls = []
    # get headers, row's name
    headers = [""]
    rows = []
    if not src:
        return result_ls
        
    for key, val in src.items():
        headers.append(key)
        for k in val:
            if not k in rows:
                rows.append(k)

    # concat list
    result_ls.append(headers)

    # rows
    for row in rows:
        content = [row]
        for header in headers:
            if header and row:
                content.append(str(src[header][row]) if row in src[header] else "")
        result_ls.append(content)
    return result_ls

def get_random_dataset(dataset: list, extract_num: int, class_name):
    np.random.shuffle(dataset)
    dataset_class = []
    for name in dataset:
        if class_name in name:
            dataset_class.append(name)
        if len(dataset_class) == extract_num:
            break
    return dataset_class

def read_json_file(path):
    with open(path, 'r') as f:
        return json.load(f)