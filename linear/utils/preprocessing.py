import csv
import datetime
import json
import logging
import os
import platform
from ast import literal_eval
from urllib import parse

import pandas as pd
import requests
from full_fred.fred import Fred

from utils.generic import verbose_func_call

# Global Variables
dm_client = None
PLATFORM = platform.processor()

os.environ["FRED_API_KEY"] = "bb1dadbc15fc4f349377288d14fb685f"


def ek_get_data(imnt, fields, format=None, con="http://localhost:8000/"):
    '''
    Takes a list<tuple<tickers, single_field>, ...> as eikon_config, returns a pandas dataframe
    >> [(List of RICs, TR.SomeField(SomeFreq), (List of RICs, TR.SomeField(SomeFreq), <...>]
    For TR field construction, refer to CODECR CodeCreator in the Eikon terminal.
    '''

    conf_dict = {
        "method": "get_data",
        "args": [imnt, fields]
    }

    # _res = pd.DataFrame(None, columns=['id', 'value', 'date'])
    # if not isinstance(eikon_config, list):
    #     raise TypeError("eikon_config must a be list<tuple<tickers, single_field>, ...>")

    _resp = requests.get(con, params=parse.urlencode(conf_dict))
    _result = json.loads(_resp.content)
    _res = pd.DataFrame(literal_eval(_result['data'].replace('null', "'null'")), columns=_result['fields'])

    if format:
        _res.columns = [i.lower().replace(' ', '_') for i in _res.columns]
        _res.loc[:, 'date'] = pd.to_datetime(_res.date).dt.date
        _res = _res.set_index('date')

    return _res


def ek_get_timeseries(imnt,
                      start_date=(datetime.datetime.today() + datetime.timedelta(-365)).strftime("%Y-%m-%d"),
                      end_date=datetime.datetime.today().strftime("%Y-%m-%d"),
                      interval='daily',
                      con="http://localhost:8000/",
                      **kwargs):
    ''' Example Config: https://docs-developers.refinitiv.com/1594387995587/14684/book/en/eikon/index.html#get_timeseries
    >> ek_get_timeseries(["USDONFSR=X"], fields="CLOSE") # U.S. LIBOR O/N
    >> ek_get_timeseries(["USDSOFR="], fields="CLOSE") # U.S. SOFR, fields="CLOSE" is optional
    '''

    # assert isinstance(imnt, list) and len(imnt) == 1, \
    #     "imnt_list should be a list type object. " \
    #     "for single imnt wrap it inside a sq bracket."

    conf_dict = {
        "method": "get_timeseries",
        "args": imnt,
        "kwargs": {
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval
        }
    }

    conf_dict['kwargs'].update(kwargs)
    _resp = requests.get(con, params=parse.urlencode(conf_dict))
    _result = json.loads(_resp.content)
    _res = pd.DataFrame(literal_eval(_result['data']), columns=_result['fields']).set_index('Date')
    return _res


def ek_get_econ(ecix_list, con="http://192.168.8.9:8000/", **kwargs):

    conf_dict = {
        "method": "get_data",
        "kwargs": {
            "instruments": ecix_list,
            "fields": ['TR.IndicatorName', 'TR.IndicatorType', 'TR.IndicatorSource', 'DSPLY_CHNM',
                       'ECI_ACT_DT', 'ACT_VAL_NS', 'FCAST_PRD', 'ECON_ACT', 'RTR_POLL',
                       'ECON_PRIOR', 'ECON_REV', 'UNIT_PREFX', 'RPT_UNITS']
        }
    }

    _resp = requests.get(con, params=parse.urlencode(conf_dict))
    _result = json.loads(_resp.content)
    _res = pd.DataFrame(literal_eval(_result['data'].replace('null', "'null'")), columns=_result['fields'])
    # _res = _res.rename(ecix_map, axis=1)
    return _res


def ek_get_analyst(imnt, raw_output=False, con="http://192.168.8.9:8000/"):
    conf_dict = {
        "method": "get_data",
        "kwargs": dict(
            instruments=imnt,
            fields=[
                # 'TR.NumOfStrongBuy(SDate=0,EDate=-144,Frq=M)', 'TR.NumOfStrongBuy(SDate=0,EDate=-144,Frq=M).calcdate',
                # 'TR.NumOfBuy(SDate=0,EDate=-144,Frq=M)', 'TR.NumOfBuy(SDate=0,EDate=-144,Frq=M).calcdate',
                # 'TR.NumOfHold(SDate=0,EDate=-144,Frq=M)', 'TR.NumOfHold(SDate=0,EDate=-144,Frq=M).calcdate',
                # 'TR.NumOfSell(SDate=0,EDate=-144,Frq=M)', 'TR.NumOfSell(SDate=0,EDate=-144,Frq=M).calcdate',
                # 'TR.NumOfStrongSell(SDate=0,EDate=-144,Frq=M)', 'TR.NumOfStrongSell(SDate=0,EDate=-144,Frq=M).calcdate',
                # 'TR.NumOfNoOpinion(SDate=0,EDate=-144,Frq=M)', 'TR.NumOfNoOpinion(SDate=0,EDate=-144,Frq=M).calcdate',
                # 'TR.NumOfRecommendations(SDate=0,EDate=-144,Frq=M)', 'TR.NumOfRecommendations(SDate=0,EDate=-144,Frq=M).calcdate',
                # 'TR.PriceTargetMean(SDate=0,EDate=-144,Frq=M)', 'TR.PriceTargetMean(SDate=0,EDate=-144,Frq=M).calcdate',
                # 'TR.Close(SDate=0,EDate=-144,Frq=M)', 'TR.Close(SDate=0,EDate=-48,Frq=M).calcdate'
                # 'TR.DilutedNormalizedEps',
                # 'TR.CoRPrimaryCountryCode',
                # 'TR.GICSSector',
                # 'TR.GICSIndustryGroup'
                # 'TR.GICSIndustry',
                # 'TR.GICSSubIndustry'
                'TR.EPSMean(SDate=0,EDate=-180,Frq=M), TR.EPSMean(SDate=0,EDate=-180,Frq=M).date',
                'TR.EPSActValue(SDate=0,EDate=-180,Frq=M), TR.EPSActValue(SDate=0,EDate=-180,Frq=M).date',
                'TR.NetProfitMean(SDate=0,EDate=-180,Frq=M), TR.NetProfitMean(SDate=0,EDate=-180,Frq=M).date',
                'TR.NetProfitActValue(SDate=0,EDate=-180,Frq=M), TR.NetProfitActValue(SDate=0,EDate=-180,Frq=M).date',
                # 'TR.Revenue',
                # 'TR.RevenueActValue',
                # 'TR.NetProfitMean(Period=FY1)',
                # 'TR.NetProfitActValue(Period=FY0)'
            ]
        )
    }

    _resp = requests.get(con, params=parse.urlencode(conf_dict))
    _result = json.loads(_resp.content)

    if raw_output:
        return _result

    _res = pd.DataFrame(literal_eval(_result['data'].replace('null', "'null'")), columns=_result['fields'])
    return _res


def fred_get(fred_ids, con=Fred()):
    _out = pd.DataFrame()

    for fred_id in fred_ids:
        tmp = con.get_series_df(fred_id)[['date', 'value']].set_index('date')
        tmp.columns = [fred_id]
        _out = _out.join(tmp, how="outer")
    return _out


@verbose_func_call
def wind_get(method, imnt, date_start, date_end, fields=None, options=None, con="http://localhost", **kwargs):
    wind_conf = {
        "method": method,
        "args": [i for i in [imnt, fields, date_start, date_end] if i is not None],
        "options": "" if options is None else options
    }

    try:
        _params = parse.urlencode(wind_conf)
        n_retries = 5
        while n_retries:
            try:
                _json = json.loads(requests.get(con, params=_params).content)
                break
            except json.JSONDecodeError:
                print(f"Method wind_get() retrying, remaining retries: {n_retries}...")
                n_retries -= 1

        if method == 'wss':
            _data = pd.DataFrame(_json['data']['Data'], index=_json['data']['Times'], columns=_json['data']['Codes']).T

        else:
            if "field_info=T" in wind_conf['options']:
                _data = pd.DataFrame(_json['data']['Data'], index=_json['data']['Fields']).T.set_index('id')
            else:
                _data = pd.DataFrame(_json['data']['Data'], index=_json['data']['Codes'], columns=_json['data']['Times']).T

    except KeyError:
        _data = None

    return _data


def bbg_get(tix, date_start, date_end, period=None, flds=None,
            non_trading_day_fill_option="NON_TRADING_WEEKDAYS",
            non_trading_day_fill_method="PREVIOUS_VALUE"):
    from utils.config import BBG_CON
    from tia.bbg import Terminal
    from tia.bbg.datamgr import BbgDataManager

    _flds = flds if flds is not None else "PX_LAST"
    dm = BbgDataManager(Terminal(*BBG_CON))
    pvt = dm[tix].get_historical(
        flds=_flds,
        start=date_start,
        end=date_end,
        period=period,
        non_trading_day_fill_option=non_trading_day_fill_option,
        non_trading_day_fill_method=non_trading_day_fill_method,
        adjustment_follow_DPDF=False,
        adjustment_split=True,
        adjustment_abnormal=True,
        adjustment_normal=True
    )
    return pvt


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
        logging.info("File not found. Cannot load from local. Use @load_local=False to proceed.")


def create_benchmark_data(tix, date_start, date_end, flds=None, save_to_local=True, source=None, **kwargs):
    """
        Downloads a list of ticker from one of 'bbg', 'eikon', 'dm', 'wind'.
        Default saving to local. Supports kwargs for some source-specific arguments.
    """
    assert source in ['bbg', 'eikon', 'wind'], "source must be either bbg, eikon OR wind"

    if source == 'bbg':
        if PLATFORM != 'arm':
            pvt = bbg_get(tix, date_start, date_end, flds=flds)
        else:
            logging.warning('blpapi not supported on Apple Silicon')
            return

    if source == 'eikon':
        _flds = flds if flds is not None else "TR.PriceClose"
        res = ek_get_data(tix,
                          [f'{_flds}(Frq=D,SDate={date_start},EDate={date_end})', f'{_flds}(Frq=D,SDate={date_start}, EDate={date_end}).date'],
                          format=True)
        pvt = res.pivot_table(index='date', columns='instrument')
        pvt.columns = pvt.columns.droplevel(0)

    if source == 'wind':
        _flds = flds if flds is not None else 'CLOSE'
        pvt = wind_get('wsd', tix, date_start, date_end, fields=_flds)
        pvt.index = pd.to_datetime(pvt.index).tz_convert(None)

    if save_to_local:
        for _col in pvt.columns:
            pvt[_col].rename('close').dropna().to_frame().to_csv(
                os.path.join("data",
                             f"{_col.split(' ')[0]}{'_' + flds.lower() if flds is not None else ''}.csv"), index=True)

    return pvt
