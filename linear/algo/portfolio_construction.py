import logging
import os
import warnings
from typing import Callable, Iterable, Union

import numpy as np
import pandas as pd

from utils.generic import verbose_func_call, ROOT_PATH
from utils.preprocessing import from_csv, create_benchmark_data

# Options
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')
# np.seterr(divide='raise')

# Global Variables
os.chdir(ROOT_PATH)

data_path = os.path.join(os.getcwd(), "data")
out_path = os.path.join(os.getcwd(), "out")


@verbose_func_call
def port_data_preprocess(symbols, date_start, date_end, load_local=True, resample_freq=None, verbose=False, **kwargs):
    '''
    Download and process price data for the used benchmarks.
    Toggle the active status in @symbols csv.
    '''

    bmk = symbols.copy()
    bmk['composite'] = bmk.bbg_code
    bmk['source'] = 'bbg'

    bmk_active = bmk.loc[bmk.asset_class != 'Unused']
    imnt_csvs = [from_csv(os.path.join(data_path, f"{comp_code}.csv")) for comp_code in bmk_active.composite]

    if verbose:
        for src, code, data in zip(bmk_active.source, bmk_active.composite, imnt_csvs):
            print(f"[Data Preprocess] Und <{code}>: @{src}\t {'on ' + data.index[-1]._date_repr if data is not None else 'does not exist'}")

    assert 'err' not in bmk_active.source, 'data source for some items are ill defined'

    # Update new assets
    updated_ser = []
    if load_local:
        px_master = pd.concat(imnt_csvs, axis=1)

    else:
        for source, code, imnt_csv in zip(bmk_active.source, bmk_active.composite, imnt_csvs):
            if imnt_csv is None:
                df = create_benchmark_data(code, date_start=date_start, date_end=date_end, save_to_local=False, source=source)
                df.index = pd.to_datetime(df.index)
                if source == 'bbg':
                    df = df.rename({'PX_LAST': code}, axis=1)

                df.to_csv(f'./data/{code}.csv')
                updated_ser.append(df)
                continue

            # if source=='bbg':
            #     imnt_csv = imnt_csv.drop('PX_LAST', axis=1).dropna()

            max_date = imnt_csv.index.max()._date_repr
            if imnt_csv.index.max() >= pd.to_datetime(date_end):
                updated_ser.append(imnt_csv)
                continue

            df_to_append = create_benchmark_data(code, date_start=max_date, date_end=date_end, save_to_local=False, source=source)

            if source == 'bbg':
                df_to_append = df_to_append.rename({'PX_LAST': code}, axis=1)

            df_to_append.index = pd.to_datetime(df_to_append.index)
            df_concat = pd.concat([imnt_csv, df_to_append], axis=0)
            df_concat = df_concat[~df_concat.index.duplicated(keep='last')]
            df_concat.to_csv(f'./data/{code}.csv')
            updated_ser.append(df_concat)

        px_master = pd.concat(updated_ser, axis=1)

    if resample_freq:
        px_master = px_master.ffill().resample(resample_freq).last()

    ret_master = px_master.pct_change().shift(-1).dropna(how='all')
    # ret_master = px_master.pct_change().replace(0, np.nan).shift(-1).dropna(how='all')
    # ret_master = px_master.pct_change().replace(0, np.nan).shift(-1).dropna(how='all')
    return bmk_active, ret_master


@verbose_func_call
def port_asset_alloc(model_name: str,
                     df_comp_phases: pd.DataFrame,
                     df_price_master: pd.DataFrame,
                     df_benchmark_metadata: pd.DataFrame,
                     model_inception_date: str,
                     weight_func: Callable,
                     roll_window: Union[int, None] = None,
                     lookback_window: Union[int, None] = None,
                     asset_dropout: Union[Iterable[str], None] = None,
                     return_hyp_weight: Union[bool, None] = True,
                     use_fallback: Union[bool, None] = False,
                     **kwargs):
    """
    Simulate portfolio allocation using the provided macro phases.
    For production strategies, refer to smartglobal_ultimax_instituition/dev-andrew repo.

    Weighting function takes 2 further argument that is @upper_bound_default and @upper_bound_dict.
    These variables can be passed as keywords, and will be propagated to the inner weight_func function.

    Args:
        model_name: name of model, not used
        df_comp_phases: phases by week, used to find associated conditional return
        df_price_master: asset return by week
        df_benchmark_metadata: underlying data
        model_inception_date: model inception date
        weight_func: weight function, should support arguments of price/roll_price/cov
        roll_window: specify return should be calculated on rolling N weeks
        lookback_window: specify return should lookback only N weeks
        asset_dropout: specify asset not used for prtf opt, for dynamic dropout without altering symbols file
        return_hyp_weight: specify if at end of model, calculate hypothetical optimal weight for all states
        use_fallback: DEPRECATED, do not use
        **kwargs: arbitrary keyword arguments, will propagate to weight_func.

    Returns:
        A list of dataframes, including backtested return, backtested weight, modelled view, and if specified, modelled optimal weight.
    """

    # try:
    if True:
        # parameters preprocess
        roll_window = 1 if roll_window is None else roll_window
        _cash = kwargs.get('cash', 1e6)
        _shift = kwargs.get('shift', 0)
        _start = kwargs.get('date_start', '2000-01-01')
        _verbose = kwargs.get('verbose', False)

        # remove unused assets
        _df_price_master = df_price_master.loc[:, ((~df_benchmark_metadata.composite.isna()) &
                                                   (df_benchmark_metadata.asset_class != 'Unused')).to_list()]

        # drop unused assets and join price master and econ phases data as master table
        price_master_prod_all = _df_price_master.drop(asset_dropout, axis=1) if asset_dropout else _df_price_master

        # truncate to correct window
        price_master_prod_all = price_master_prod_all.loc[price_master_prod_all.index >= _start]
        price_master_roll_all = ((1 + price_master_prod_all)[::-1].rolling(roll_window, min_periods=1).apply(np.prod) - 1)

        pvt_return_ts = pd.DataFrame()
        pvt_weight_ts = pd.DataFrame(columns=df_price_master.columns)
        pvt_bl_return = pd.DataFrame() #columns=df_benchmark_metadata.asset_class.unique())
        lst_tradedate = price_master_prod_all.index[(price_master_prod_all.index >= model_inception_date)].to_list()

        if df_comp_phases.index.max() > lst_tradedate[-1]:
            lst_tradedate.append(df_price_master.index[-1] + pd.tseries.offsets.MonthBegin(1))

        # get each SOM
        lst_rebalance = pd.to_datetime(pd.Series(lst_tradedate).apply(lambda x: f"{x.year:04d}-{x.month:02d}-01").unique()).to_list()

        # incrementally train model
        for mdl_dt in lst_tradedate:

            if mdl_dt in df_comp_phases.index:
                current_phase = df_comp_phases.loc[mdl_dt].macro_phase.item()
            else:
                # if for some reason we dont have longer projection than our backtesting, use the last phase.
                # otherwise, locate the next phase immediately after backtesting period, and output the associated
                # BL view.
                proj = df_comp_phases.loc[mdl_dt + pd.tseries.offsets.Day(1):]
                if len(proj):
                    current_phase = proj.iloc[0].macro_phase.item()
                else:
                    current_phase = df_comp_phases.iloc[-1].macro_phase.item()

            price_master_prod = price_master_prod_all.loc[price_master_prod_all.index < mdl_dt]
            price_master_roll = price_master_roll_all.loc[price_master_roll_all.index < mdl_dt]

            if lookback_window:
                assert lookback_window > 0, "lookback should be strictly positive and in n_periods of the sampling frequency."
                price_master_prod = price_master_prod.iloc[-lookback_window:]
                price_master_roll = price_master_roll.iloc[-lookback_window:]

            # calculate historical metrics
            phase_loc = df_comp_phases[(df_comp_phases.macro_phase == current_phase) &
                                       (df_comp_phases.index < mdl_dt)].index

            if use_fallback:
                phase_loc_fallback = df_comp_phases[
                    (df_comp_phases.macro_phase == (
                                current_phase + df_comp_phases.econ_phase.max()) % df_comp_phases.macro_phase.max()) &
                    (df_comp_phases.index < mdl_dt)
                    ].index
            else:
                phase_loc_fallback = pd.core.indexes.datetimes.DatetimeIndex([])

            perf_by_phase = price_master_prod.loc[price_master_prod.index.intersection(phase_loc)]
            perf_by_phase_roll = price_master_roll.loc[price_master_prod.index.intersection(phase_loc)]

            if len(perf_by_phase) < len(price_master_prod) * 0.02:
                perf_by_phase_fallback = price_master_prod.loc[price_master_prod.index.intersection(phase_loc_fallback)]
                perf_by_phase = pd.concat([perf_by_phase, perf_by_phase_fallback])

                perf_by_phase_roll_fallback = price_master_roll.loc[
                    price_master_prod.index.intersection(phase_loc_fallback)]
                perf_by_phase_roll = pd.concat([perf_by_phase_roll, perf_by_phase_roll_fallback])

            if len(lst_rebalance):
                if mdl_dt >= lst_rebalance[0]:
                    rebal_dt = lst_rebalance.pop(0)

                    if _verbose:
                        print(f"Rebalancing on {rebal_dt=}, {current_phase=}")

                    # perform weighting if phase data is sufficient
                    if len(perf_by_phase) >= 2:
                        n_retries = 3
                        while n_retries:
                            try:
                                opt_weight, opt_res = weight_func(
                                    date=mdl_dt,
                                    price=perf_by_phase,
                                    roll_price=perf_by_phase_roll,
                                    cov_base=price_master_prod.iloc[-52 * 3:],
                                    **kwargs
                                )
                                assert opt_res.success, f"[port_asset_alloc] Optimization failed in {weight_func}."
                            except AssertionError:
                                print(f"[port_asset_alloc] Retrying optimization in {weight_func} on date={mdl_dt}. {n_retries} retries remaining.")
                                opt_weight = {i: (x / opt_res.x.sum()) for i, x in zip(opt_weight.keys(), opt_res.x)}
                                n_retries -= 1
                                continue
                            break
                    else:
                        try:
                            len(opt_weight)
                        except UnboundLocalError:
                            opt_weight = {i: (1 / len(perf_by_phase.columns)) for i in perf_by_phase.columns}
                        if _verbose:
                            print(f"[port_asset_alloc] weighting skipped on {mdl_dt}")

                    pvt_weight = pd.Series(opt_weight, name="weight")
                    assert np.isclose(pvt_weight.sum().item(), 1.)
                    cash_weight = _cash * pvt_weight

            # aggregate output := 1/ Asset Class + 2/ Und Level
            r = pd.concat([
                (((1 + perf_by_phase_roll) ** (1 / roll_window) - 1)
                 .rename(df_benchmark_metadata.set_index('composite').asset_class.to_dict(), axis=1)
                 .groupby(level=0, axis=1)
                 .mean()
                 .mean()),
                ((1 + perf_by_phase_roll) ** (1 / roll_window) - 1).mean()
            ])
            pvt_bl_return.loc[mdl_dt, r.index] = r

            if mdl_dt in price_master_prod_all.index:
                pvt_return_ts.loc[mdl_dt, 'return'] = np.nansum(price_master_prod_all.loc[mdl_dt] * cash_weight)
                pvt_weight_ts.loc[mdl_dt] = cash_weight
                cash_weight = (1+price_master_prod_all.loc[mdl_dt]) * cash_weight
                _cash = np.sum(cash_weight)

        if return_hyp_weight:
            hyp_optimal_weight = {}

            for phase in df_comp_phases.macro_phase.unique():
                phase_loc = df_comp_phases[(df_comp_phases.macro_phase == phase)].index

                try:
                    hyp_weight, hyp_res = weight_func(
                        date=mdl_dt,
                        price=price_master_prod.loc[price_master_prod.index.intersection(phase_loc)],
                        roll_price=price_master_roll.loc[price_master_prod.index.intersection(phase_loc)],
                        cov_base=price_master_prod.iloc[-52 * 3:],
                        **kwargs
                    )
                except Exception as e:
                    logging.warning(f"On calculate hypothetical optimal weight, caught {e.__class__.__name__}: {e.args}")
                    hyp_weight = {k: 0 for k in price_master_prod.columns}

                hyp_optimal_weight[phase] = hyp_weight

            pvt_hyp_optimal_weight = pd.DataFrame(hyp_optimal_weight).sort_index(axis=1)

        else:
            pvt_hyp_optimal_weight = None

        return pvt_return_ts, pvt_weight_ts, pvt_bl_return, pvt_hyp_optimal_weight

@verbose_func_call
def port_backtest(model_name, pretrained_weight, df_price_master, date_start, date_end, base_cash=1e6, verbose=False,
                  **kwargs):

    # date_start = date_start or pretrained_weight.index[0].strftime("%Y-%m-%d")

    # parameters preprocess
    pvt_return_ts = pd.DataFrame()
    pvt_weight_ts = pd.DataFrame(columns=df_price_master.columns)
    pvt_txcost_ts = pd.Series(dtype=float)
    lst_tradedate = df_price_master.index[(df_price_master.index >= date_start) &
                                          (df_price_master.index <= date_end)].to_list()
    lst_rebalance = pretrained_weight.index.to_list()

    # for backtesting only.
    if pretrained_weight is None:
        raise Exception("pretrained_weight must not be empty")

    cash = base_cash
    cash_weight = None

    for mdl_dt in lst_tradedate:
        if len(lst_rebalance):
            if mdl_dt >= lst_rebalance[0] or cash_weight is None:
                if verbose:
                    print(f"Rebalance at {mdl_dt._date_repr} **using pretrained weight**")

                lst_rebalance.pop(0)
                pvt_weight = pretrained_weight.loc[pretrained_weight[(pretrained_weight.index.year == mdl_dt.year) & (pretrained_weight.index.month == mdl_dt.month)].index[0]]
                assert np.isclose(pvt_weight.sum(), 1), f"weight on {mdl_dt._date_repr} does not sum to 1, got {pvt_weight.sum()}"
                cash_weight = cash * pvt_weight
                if len(pvt_weight_ts):
                    pvt_txcost_ts.loc[mdl_dt] = np.sum(np.abs(cash_weight - pvt_weight_ts.iloc[-1])) * 0.001
                else:
                    pvt_txcost_ts.loc[mdl_dt] = np.sum(cash_weight) * 0.001

        # aggregate output
        pvt_return_ts.loc[mdl_dt, 'return'] = np.nansum(df_price_master.loc[mdl_dt] * cash_weight)
        # pvt_weight_ts = pd.concat([pvt_weight_ts, cash_weight], axis=0)
        pvt_weight_ts.loc[mdl_dt] = cash_weight
        cash_weight = (1 + df_price_master.loc[mdl_dt]) * cash_weight
        cash = np.sum(cash_weight)

    pvt_return_ts_after_txcost = pd.concat([pvt_return_ts, -pvt_txcost_ts], axis=1).fillna(0).sum(axis=1).rename('return').to_frame()
    return pvt_return_ts, pvt_weight_ts, pvt_return_ts_after_txcost


def compute_win_rate(asset_perf, comp_phases, benchmark_metadata, bl_views, rolling_period=2):
    df = ((1 + asset_perf)[::-1].rolling(rolling_period, min_periods=1).apply(np.prod) - 1)[::-1]
    df2 = df.join(comp_phases.reindex(asset_perf.index, method='ffill').macro_phase)
    df2 = df2.loc[~pd.isna(df2.macro_phase), :]
    df3 = df2.copy()
    df3.columns = df3.columns.map(
        benchmark_metadata.set_index('composite').asset_class.to_dict() | {'macro_phase': 'macro_phase'})
    df3 = df3.groupby(level=0, axis=1).mean()  # .replace(0, np.nan)

    # compute t-stat (mean/std) table
    t_stat = (
        (52 / rolling_period * df3.groupby('macro_phase').mean()).applymap(lambda x: f"{np.round(x, 4):.4f}") + ' (' +
        (np.sqrt(52 / rolling_period) * df3.groupby('macro_phase').std()).applymap(lambda x: f"{np.round(x, 4):.4f}") + ')'
    )

    bl = bl_views.reindex(df3.columns, axis=1).sort_index(level=0, axis=1).drop('macro_phase', axis=1)
    cmp = df3.reindex(bl.index).drop('macro_phase', axis=1)
    sgn = (np.sign(bl) == np.sign(cmp))
    win_rate = sgn.sum(axis=0) / sgn.count(axis=0)
    win_rate_avg = win_rate.values.prod() ** (1 / win_rate.shape[0])

    print("[ T-Stats ]")
    print(t_stat.to_markdown(stralign='right'), '\n')

    print("[ Win Rate ]")
    print(win_rate)
    print(f"\nwin rate avg: {win_rate_avg:.6f}")
