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

from typing import List

os.environ["FRED_API_KEY"] = "bb1dadbc15fc4f349377288d14fb685f"
FRED_CON = None


def fred_get_metadata(ser: str, con=FRED_CON) -> str:
    """ Takes one single fred id, return a single fred id metadata string. """
    if con is None:
        FRED_CON = Fred()
        con = FRED_CON

    res = con.get_tags_for_a_series(ser)
    return ' | '.join([i['notes'] for i in res['tags'] if i['notes']])


def fred_get(fred_ids: List[str], con=FRED_CON) -> pd.DataFrame:
    """ Takes a list of fred ids, returns a pandas DataFrame. """
    if con is None:
        FRED_CON = Fred()
        con = FRED_CON

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
