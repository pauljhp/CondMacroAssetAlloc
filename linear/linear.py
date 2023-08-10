import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algo.optimization import weight_func_min_var, max_sharpe_cvxpy
from utils.analytics import create_analytics
from utils.preprocessing import from_csv
from utils.search import ensemble_create_cases, ensemble_execute, ensemble_get_weight_config_view

from algo.portfolio_construction import port_data_preprocess, port_asset_alloc, port_backtest, compute_win_rate


# Global Variables
...

# Options
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')


def main(config_, load_local=True):
    date_config = config_

    # 1/ Read data
    gdp = from_csv('./inp/gdp_zscore.csv')
    cpi = from_csv('./inp/cpi_zscore.csv')
    fci = from_csv('./inp/fci_zscore_macro.csv')

    # 2/ Compose macro data
    from utils.transform import savgol_pd, normalize
    savgol = lambda x: savgol_pd(x, 12, 3, 0)

    compo = pd.concat([
        savgol(normalize(gdp.diff(3))[0]),
        savgol(normalize(cpi.diff(3))[0]),
        savgol(normalize(fci)[0]),
    ], axis=1)
    compo.columns = ['economic', 'inflation', 'financial_condition']

    # 3/ Backtest Data Preparation
    df_symbols = pd.read_csv('underlyings.csv')
    df_benchmark_metadata, df_asset_perf = port_data_preprocess(
        symbols=df_symbols,
        load_local=load_local,
        verbose=True,
        # date_start='2000-01-01',
        # date_end='2023-03-31',
        **date_config
    )
    df_asset_perf = df_asset_perf.loc[:date_config['date_end']]

    # 4/ Perform Backtest
    thres_ = {'economic': -0.1, 'inflation': -0.2, 'financial_condition': 0.3}

    print(f"{thres_=}")
    compo_ = compo.ffill().copy()

    for k in compo_.columns:
        compo_.loc[:, k] = compo_.loc[:, k].apply(lambda x: "1" if x > thres_[k] else "0")

    # 3-phase ML clock
    if 'rates' in compo_.columns:
        compo_.loc[(compo_.economic == '0') & (compo_.inflation == '0') & (compo_.rates == '0'), 'state'] = 0
        compo_.loc[(compo_.economic == '1') & (compo_.inflation == '0') & (compo_.rates == '0'), 'state'] = 1
        compo_.loc[(compo_.economic == '1') & (compo_.inflation == '1') & (compo_.rates == '0'), 'state'] = 2
        compo_.loc[(compo_.economic == '0') & (compo_.inflation == '1') & (compo_.rates == '0'), 'state'] = 3
        compo_.loc[(compo_.economic == '0') & (compo_.inflation == '0') & (compo_.rates == '1'), 'state'] = 4
        compo_.loc[(compo_.economic == '1') & (compo_.inflation == '0') & (compo_.rates == '1'), 'state'] = 5
        compo_.loc[(compo_.economic == '1') & (compo_.inflation == '1') & (compo_.rates == '1'), 'state'] = 6
        compo_.loc[(compo_.economic == '0') & (compo_.inflation == '1') & (compo_.rates == '1'), 'state'] = 7

    if 'financial_condition' in compo_.columns:
        compo_.loc[(compo_.economic == '0') & (compo_.inflation == '0') & (compo_.financial_condition == '0'), 'state'] = 0
        compo_.loc[(compo_.economic == '1') & (compo_.inflation == '0') & (compo_.financial_condition == '0'), 'state'] = 1
        compo_.loc[(compo_.economic == '1') & (compo_.inflation == '1') & (compo_.financial_condition == '0'), 'state'] = 2
        compo_.loc[(compo_.economic == '0') & (compo_.inflation == '1') & (compo_.financial_condition == '0'), 'state'] = 3
        compo_.loc[(compo_.economic == '0') & (compo_.inflation == '0') & (compo_.financial_condition == '1'), 'state'] = 4
        compo_.loc[(compo_.economic == '1') & (compo_.inflation == '0') & (compo_.financial_condition == '1'), 'state'] = 5
        compo_.loc[(compo_.economic == '1') & (compo_.inflation == '1') & (compo_.financial_condition == '1'), 'state'] = 6
        compo_.loc[(compo_.economic == '0') & (compo_.inflation == '1') & (compo_.financial_condition == '1'), 'state'] = 7

    state = compo_.state
    df_comp_phases = state.rename('macro_phase').to_frame().resample('W').ffill().loc["2001-01-01":].astype(int)

    # portfolio preprocess and optimization
    e_cases = ensemble_create_cases(
        {
            # "weight_func": [weight_func_min_var, max_sharpe_cvxpy],
            "roll_window": [4, 8, 12, 18, 24],
            "dp": [1.5, 2., 2.5, 3.0],
        }
    )
    #
    # for case in e_cases:
    #     res, wgt, *_ = port_asset_alloc(
    #         model_name="Macro Model",
    #         df_comp_phases=df_comp_phases,
    #         df_price_master=df_asset_perf,
    #         df_benchmark_metadata=df_benchmark_metadata,
    #         use_fallback=False,
    #         weight_func=max_sharpe_cvxpy,
    #         cases=e_cases,
    #         upper_bound_default=0.8,
    #         upper_bound_dict={"VNQ US Equity": 0.25, "DBC US Equity": 0.25, "IAU US Equity": 0.25},
    #         verbose=False,
    #         **date_config,
    #         **case,
    #     )

    e_res = ensemble_execute(
        func=port_asset_alloc,
        model_name="Macro Model",
        df_comp_phases=df_comp_phases,
        df_price_master=df_asset_perf,
        df_benchmark_metadata=df_benchmark_metadata,
        use_fallback=False,
        weight_func=max_sharpe_cvxpy,
        cases=e_cases,
        upper_bound_default=0.8,
        upper_bound_dict={"VNQ US Equity": 0.25, "DBC US Equity": 0.25, "IAU US Equity": 0.25},
        verbose=False,
        **date_config,
    )

    # extract and backtest the mixed portfolio
    N_TOP = 5
    pretrained_weight, top_configs, bl_views, optimal_weight = ensemble_get_weight_config_view(e_res, e_cases, n_top=N_TOP)

    # print out configs
    for top_config in top_configs:
        print(*top_config)

    # retrieve optimal weight
    nm = df_symbols.set_index('bbg_code').en_desc
    optimal_weight.T.rename(nm, axis=1).plot.bar(stacked=True, title='Optimal weight under each phase')
    plt.tight_layout()
    plt.show()

    # backtest
    _, df_asset_perf_backtest = port_data_preprocess(
        symbols=df_symbols,
        load_local=load_local,
        verbose=False,
        date_start='2000-01-01',
        date_end='2023-03-31',
        # **date_config
    )

    pwgt_ms = pretrained_weight.resample('MS').first()
    ret, wgt, ret_tx = port_backtest(
        model_name="top_mixed_weight",
        date_start='2011-01-01',
        date_end=date_config['date_end'],
        df_price_master=df_asset_perf_backtest,
        pretrained_weight=pwgt_ms,
    )

    # plot
    fig, ax = plt.subplots(figsize=(10, 4))
    wgt.rename(nm, axis=1).clip(0).plot.area(ax=ax, stacked=True, cmap='viridis', title="Backtesting return (starting NAV=$1 mio)")
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=5)
    plt.tight_layout()
    plt.show()

    # win rate
    compute_win_rate(
        asset_perf=df_asset_perf,
        comp_phases=df_comp_phases,
        benchmark_metadata=df_benchmark_metadata,
        bl_views=bl_views
    )

    # analytics
    analytics = create_analytics([ret, ], ).T
    print("========== Monthly Rebalanced ==========")
    print(analytics.to_csv())

    pwgt_qs = pretrained_weight.resample('QS').first()
    ret_qs, wgt_qs, _ = port_backtest(
        model_name="top_mixed_weight",
        date_start='2011-01-01',
        date_end=date_config['date_end'],
        df_price_master=df_asset_perf_backtest,
        pretrained_weight=pwgt_qs,
    )

    # analytics
    print("========== Quarterly Rebalanced ==========")
    analytics = create_analytics([ret_qs, ], ).T
    print(analytics.to_csv())

    # compute turnover
    ax = ret.cumsum().rename(lambda x: 'Monthly rebal - monthly', axis=1).plot(title='Strategy $ cumulative return / turnover', figsize=(10, 4))
    turnover = (wgt.diff().fillna(wgt).resample('MS').first().abs().sum(axis=1) / wgt.resample('MS').first().sum(axis=1)).cumsum().rename('Monthly rebal - turnover').to_frame()
    turnover.plot(ax=ax, secondary_y=True)
    ret_qs.cumsum().rename(lambda x: 'Quarterly rebal - return', axis=1).plot(ax=ax)
    (wgt.diff().fillna(wgt).resample('QS').first().abs().sum(axis=1) / wgt.resample('QS').first().sum(axis=1)).cumsum().rename('Quarterly rebal - turnover').to_frame().plot(ax=ax, secondary_y=True)
    plt.tight_layout()
    plt.show()

    # outperformance
    fig, bx = plt.subplots(figsize=(10, 4))
    op = (
        (1 + (1e6 + ret.cumsum()).resample('MS').last().pct_change()) /
        (1 + (1e6 + ret_qs.cumsum()).resample('MS').last().pct_change())
    ).fillna(1.).squeeze() - 1
    op.plot.line(title='Outperformance of monthly vs quarterly', ax=bx, zorder=1, linestyle='None')
    bx.set_ylim(-0.1, 0.1)
    bx.fill_between(op.index, op.clip(-1, 0), 0, color='red', zorder=0)
    bx.fill_between(op.index, 0, op.clip(0, 1), color='green', zorder=0)
    plt.grid()
    plt.tight_layout()
    plt.show()


def driver():
    os.makedirs('./research/', mode=0o755, exist_ok=True)
    os.makedirs('./research/out/', mode=0o755, exist_ok=True)
    os.makedirs('./data/', mode=0o755, exist_ok=True)

    config = {
        "date_start": "2000-01-01",
        "date_end": "2023-03-31",
        "resample_freq": "W",
        "model_inception_date": "2011-01-01"
    }

    main(config, load_local=True)


if __name__ == "__main__":
    driver()
