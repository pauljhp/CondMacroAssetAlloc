import csv
import datetime
import json
import logging
import os
from ast import literal_eval
from urllib import parse
from zipfile import ZipFile
from full_fred.fred import Fred
import pandas as pd
import requests

os.environ["FRED_API_KEY"] = "bb1dadbc15fc4f349377288d14fb685f"


def fred_get(fred_ids, con=Fred()):
    _out = pd.DataFrame()

    for fred_id in fred_ids:
        tmp = con.get_series_df(fred_id)[['date', 'value']].set_index('date')
        tmp.columns = [fred_id]
        _out = _out.join(tmp, how="outer")
    return _out


def to_csv(export_path, list_of_data, field_names):
    with open(export_path, 'w') as f:
        _writer = csv.writer(f)
        _writer.writerow(field_names)
        _writer.writerows(list_of_data)
    print(f"[Data] Successfully written to {export_path}.")


def from_csv(import_path, **kwargs):
    try:
        return pd.read_csv(import_path, index_col=0, parse_dates=True, infer_datetime_format=True, **kwargs)
    except FileNotFoundError:
        logging.info("File not found.")


if __name__ == '__main__':
    ...
