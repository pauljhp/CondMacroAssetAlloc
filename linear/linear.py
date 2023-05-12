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
            "weight_func": [weight_func_min_var, max_sharpe_cvxpy],
            "roll_window": [4, 8, 12, 18, 24],
            "dp": [1.5, 2., 2.5, 3.0],
        }
    )

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
        upper_bound_dict={"VNQ": 0.25, "DBC US Equity": 0.25, "IAU": 0.25},
        verbose=False,
        **date_config,
    )

    # extract and backtest the mixed portfolio
    N_TOP = 5
    pretrained_weight, top_configs, bl_views, optimal_weight = ensemble_get_weight_config_view(e_res, e_cases, n_top=N_TOP)

    views_by_phase = pd.concat([bl_views, df_comp_phases], axis=1).resample('MS').first().dropna().apply(np.round, decimals=6)
    views_by_phase.to_csv(os.path.join('research', 'out', 'bl_views.csv'), index=True)

    # print out configs
    for top_config in top_configs:
        print(*top_config)

    # backtest
    ret, wgt, ret_tx = port_backtest(
        model_name="top_mixed_weight",
        date_start='2011-01-01',
        date_end=date_config['date_end'],
        df_price_master=df_asset_perf,
        pretrained_weight=pretrained_weight,
    )

    # plot
    fig, ax = plt.subplots(figsize=(10, 4))
    wgt.clip(0).plot.area(ax=ax, stacked=True, cmap='viridis')
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=5)
    plt.tight_layout()
    plt.show()

    # analytics
    analytics = create_analytics([ret, ], ).T
    print(analytics.to_csv())

    # win rate
    compute_win_rate(
        asset_perf=df_asset_perf,
        comp_phases=df_comp_phases,
        benchmark_metadata=df_benchmark_metadata,
        bl_views=bl_views
    )

    # data dump
    compo.to_csv('./research/out/model_display_us_4a_viz_signals.csv')
    df_comp_phases.resample('MS').first().to_csv('./research/out/model_display_us_4b_viz_phases.csv')

    # (df_comp_phases.squeeze().apply(int).value_counts() / len(df_comp_phases)).apply(lambda x: f"{x:.2%}")
    tbl = compo_.reset_index().drop('index', axis=1)
    tally = tbl.groupby('state').min().applymap(lambda x: {'0': '-', '1': '+'}[x])
    tally.loc[:, 'state_count'] = (
        tally.index.map(int).map(
            (df_comp_phases.squeeze().apply(int).value_counts() / len(df_comp_phases)).apply(lambda x: f"{x:.2%}")
        )
    )
    tally.to_csv('./research/out/model_display_us_4c_viz_phase_description.csv')

    N_ROLL = 4
    df = ((1 + df_asset_perf)[::-1].rolling(N_ROLL, min_periods=1).apply(np.prod) - 1)[::-1]
    df2 = df.join(df_comp_phases.reindex(df_asset_perf.index, method='ffill').macro_phase)
    df2 = df2.loc[~pd.isna(df2.macro_phase), :]
    df3 = df2.copy()
    df3.columns = df3.columns.map(
        df_benchmark_metadata.set_index('composite').asset_class.to_dict() | {'macro_phase': 'macro_phase'})
    df3 = df3.groupby(level=0, axis=1).mean()
    df_mean = 52 / N_ROLL * df3.groupby('macro_phase').mean()
    df_std = np.sqrt(52 / N_ROLL) * df3.groupby('macro_phase').std()
    df_sharpe = df_mean / df_std

    df_mean.to_csv('./research/out/model_display_us_5a_viz_asset_return_mean.csv')
    df_sharpe.to_csv('./research/out/model_display_us_5b_viz_asset_return_sharpe.csv')
    optimal_weight.to_csv('./research/out/model_display_us_5c_viz_optimal_weight.csv')

    ret.to_csv('./research/out/model_display_us_6a_viz_backtest_ret.csv')
    wgt.to_csv('./research/out/model_display_us_6a_viz_backtest_weight.csv')
    analytics.to_csv('./research/out/model_display_us_6b_viz_backtest_metrics.csv')


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
