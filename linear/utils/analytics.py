import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


def mdd(arr, base_cash=1e6):
    _arr = (1 + (arr / base_cash).cumsum()).pct_change().dropna().values
    foo, bar = _arr[0], 0
    for ele in _arr:
        bar = (1 + bar) * (1 + ele) - 1
        if bar > 0:
            bar = 0
        elif foo > bar:
            foo = bar
    return foo.item()


def create_analytics(cash_return,
                     timeframe='full',
                     rename_to=None,
                     as_dataframe=True,
                     base_cash=1e6):
    assert (isinstance(cash_return, list) or isinstance(cash_return, tuple)), \
        "cash_return must be a list. wrap a single series into a one-item list."

    if not rename_to:
        rename_to = list(range(len(cash_return)))

    assert (isinstance(rename_to, list) or isinstance(rename_to, tuple)), \
        "rename_to must be a list. wrap a single series into a one-item list."
    assert len(cash_return) == len(rename_to), \
        "length of cash_return and rename_to should be equal."

    ref_date = np.min([ser.index[-1] for ser in cash_return])
    ytd_days_elapsed = ref_date.day_of_year
    soy_date = ref_date + relativedelta(days=-ytd_days_elapsed)

    dates = {
        "full": {"date_start": None, "date_end": None},
        "5y": {"date_start": ref_date + relativedelta(years=-5), "date_end": None},
        "3y": {"date_start": ref_date + relativedelta(years=-3), "date_end": None},
        "2y": {"date_start": ref_date + relativedelta(years=-2), "date_end": None},
        "1y": {"date_start": ref_date + relativedelta(years=-1), "date_end": None},
        "ytd": {"date_start": soy_date, "date_end": None},
    }

    for year in range(2010, 2023):
        dates.update({
            str(year): {"date_start": pd.to_datetime(f"{str(year-1)}-12-31"),
                        "date_end": pd.to_datetime(f"{str(year)}-12-31")}
        })

    assert timeframe in dates.keys(), f"invalid @timeframe. you can use one of {dates.keys()}"

    date_start, date_end = dates[timeframe]['date_start'], dates[timeframe]['date_end']
    analytics = {}

    for ret_full, name in zip(cash_return, rename_to):
        # ret = ret_full[ret_full['return'] != 0][date_start:date_end]
        _ser = (ret_full[ret_full['return'] != 0]).cumsum()
        ret = (((_ser + base_cash) / (_ser + base_cash).loc[date_start:].iloc[0])[
               date_start:date_end] * base_cash - base_cash).diff()
        _ret, *_ = (1 + (ret / base_cash).cumsum().iloc[-1] - 1).values
        _std, *_ = (
            (1 + (ret / base_cash).cumsum()).pct_change().std().values
            * np.sqrt(np.round(365 / ((ret_full.index[[0, -1]][1] - ret_full.index[[0, -1]][0]).days / len(ret_full))))
        )
        _tmd = (ret.index[-1] - ret.index[0]).days / 365.25
        _rpa = ((1 + _ret) ** (1 / _tmd) - 1)
        _shp = _rpa / _std
        _mdd = mdd(ret, base_cash=base_cash)
        _cmr = -_rpa / _mdd

        analytics.update(
            {
                name: {
                    "total_return": _ret,
                    "annu_return": _rpa,
                    "annu_volatility": _std,
                    "sharpe_ratio": _shp,
                    "maximum_drawdown": _mdd,
                    "calmar_ratio": _cmr
                }
            }
        )

    df_body = pd.DataFrame(analytics).T.loc[:, ["total_return", "annu_return", "annu_volatility", "sharpe_ratio", "maximum_drawdown", "calmar_ratio"]]

    if as_dataframe:
        return df_body

    return analytics


if __name__ == "__main__":
    ...
