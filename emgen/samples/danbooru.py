import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


def read_json(json_file):
    json_dicts = []
    with open(json_file, 'r') as jf:
        for line in jf:
            json_line = json.loads(line)
            json_dicts.append(json_line)
    return json_dicts


def filter_json(json_dicts, *args, **kwargs):
    # *args are functions run on a single dict that return
    # a bool for whether or not to include
    # **kwargs are direct filters on JSON keys/values
    json_filtered = json_dicts
    for k in kwargs.keys():
        json_filtered = [json_dict for json_dict in json_filtered
                         if json_dict[k] == kwargs[k]]
    for filter in args:
        json_filtered = [json_dict for json_dict in json_filtered
                         if filter(json_dict)]
    return json_filtered


def extract_file(file_name,
                 zip_location='./data/db/db_dataset.zip',
                 extract_location='./data/db'):
    file_name = Path(file_name)
    zip_location = Path(zip_location)
    extract_location = Path(extract_location)
    with ZipFile(zip_location, 'r') as zip:
        json_list = zip.read(str(file_name))
    with open(extract_location / file_name.name, 'wb') as f:
        f.write(json_list)


def format_workdir(json_dicts,
                   zip_location='./data/db/db_dataset.zip',
                   images_location='danbooru-images/danbooru-images',
                   extract_location='./data/db'):
    # Outputs images in requested dir and produces csv
    images_location = Path(images_location)
    extract_location = Path(extract_location)
    names = [images_location
             / f'{int(json_dict["id"]) % 1000 :04}/{json_dict["id"]}.jpg'
             for json_dict in json_dicts]
    uploaders = [int(json_dict['uploader_id']) for json_dict in json_dicts]

    # Create csv
    names_heads = [Path(name).name for name in names]  # NO
    (pd.DataFrame({'file_name': names_heads, 'label': uploaders})
        .to_csv(extract_location / 'labels.csv', index=False))

    # Extract images
    with ZipFile(zip_location, 'r') as zip:
        for name in names:
            img = zip.read(str(name))
            # img = cv2.imdecode(np.frombuffer(img, np.uint8),
            #                    cv2.IMREAD_COLOR)
            # cv2.imwrite(extract_location / Path(name).name, img)
            with open(extract_location / Path(name).name, 'wb') as f:  # NO
                f.write(img)
